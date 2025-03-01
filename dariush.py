import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer as HFTokenizer
from tqdm import tqdm
import random
import os

# 1. پیکربندی پیشرفته
class DariushConfig:
    def __init__(self):
        self.vocab_size = 50000
        self.emb_size = 512
        self.num_heads = 8
        self.num_layers = 6
        self.hidden_size = 2048
        self.max_seq_len = 1024
        self.num_experts = 4
        self.top_k = 2
        self.batch_size = self._dynamic_batch_size()
        self.learning_rate = 2e-5
        self.num_epochs = 10
        self.dropout = 0.1
        self.device = self._detect_device()
        self.use_amp = self.device.type != "cpu"
        self.special_tokens = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[CLS]": 4, "[SEP]": 5}

    def _detect_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _dynamic_batch_size(self):
        if self.device.type == "cuda":
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return min(16, int(mem * 2))
        return 4

config = DariushConfig()

# 2. توکنایزر
class PersianTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE())
        self.special_tokens = config.special_tokens

    def train(self):
        dataset = load_dataset("oscar", "unshuffled_deduplicated_fa", split="train[:5%]")
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=config.vocab_size, special_tokens=list(self.special_tokens.keys()))
        self.tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
        self.tokenizer.save("tokenizer.json")

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

# 3. دیتاست
class PersianDataset(Dataset):
    def __init__(self, tokenizer, split="train"):
        self.tokenizer = tokenizer
        self.data = load_dataset("oscar", "unshuffled_deduplicated_fa", split=split[:1000])["text"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer.encode(text)[:config.max_seq_len]
        input_ids = tokens[:-1]
        labels = tokens[1:]
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.ones_like(torch.tensor(input_ids))
        }

# 4. مدل DariushGPT
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, x, seq_len):
        pos = torch.arange(seq_len, device=x.device).float()
        rotary_emb = pos[:, None] * self.inv_freq[None, :]
        return rotary_emb

    def apply_rotary(self, x, rotary_emb):
        B, S, H, D = x.shape
        sin_val = torch.sin(rotary_emb).to(x.device)
        cos_val = torch.cos(rotary_emb).to(x.device)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x[..., 0::2] = x1 * cos_val - x2 * sin_val
        x[..., 1::2] = x1 * sin_val + x2 * cos_val
        return x

class MoE(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_size, num_experts)
        self.top_k = top_k

    def forward(self, x):
        gates = torch.softmax(self.gate(x), dim=-1)
        top_gates, top_indices = torch.topk(gates, k=self.top_k, dim=-1)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=2)
        selected_outputs = expert_outputs.gather(2, top_indices.unsqueeze(-1).expand_as(expert_outputs))
        return (selected_outputs * top_gates.unsqueeze(-1)).sum(dim=2)

class EnhancedTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb_size = config.emb_size
        self.num_heads = config.num_heads
        self.head_dim = self.emb_size // self.num_heads
        self.attn = nn.MultiheadAttention(self.emb_size, self.num_heads, dropout=config.dropout)
        self.moe = MoE(config.hidden_size, config.num_experts, config.top_k)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        self.norm1 = nn.LayerNorm(self.emb_size)
        self.norm2 = nn.LayerNorm(self.emb_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        B, S, _ = x.shape
        rotary_emb = self.rotary_emb(x, seq_len=S)
        q = k = v = x.view(B, S, self.num_heads, self.head_dim)
        q = self.rotary_emb.apply_rotary(q, rotary_emb)
        k = self.rotary_emb.apply_rotary(k, rotary_emb)
        q, k, v = q.view(B, S, self.emb_size), k.view(B, S, self.emb_size), v.view(B, S, self.emb_size)
        attn_output, _ = self.attn(q, k, v, attn_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        moe_output = self.moe(x)
        return self.norm2(x + self.dropout(moe_output))

class DariushGPT(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_size)
        self.layers = nn.ModuleList([EnhancedTransformerLayer(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.emb_size)
        self.output = nn.Linear(config.emb_size, config.vocab_size)
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        return self.output(x)

# 5. آموزش (Trainer)
class DariushTrainer:
    def __init__(self, model, tokenizer):
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.scaler = GradScaler(enabled=config.use_amp)

    def train(self, dataset):
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        for epoch in range(config.num_epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
                inputs = batch["input_ids"].to(config.device)
                masks = batch["attention_mask"].to(config.device)
                labels = batch["labels"].to(config.device)

                self.optimizer.zero_grad()
                with autocast(enabled=config.use_amp):
                    outputs = self.model(inputs, masks)
                    loss = F.cross_entropy(outputs.view(-1, config.vocab_size), labels.view(-1))
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")
            torch.save(self.model.state_dict(), "dariush.pth")

# 6. قابلیت‌های جدید
class QAModel:
    def __init__(self, model_name="HooshvareLab/bert-fa-base-uncased"):
        self.tokenizer = HFTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def answer(self, context, question):
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        return self.tokenizer.decode(inputs["input_ids"][0][start:end])

class PoetryGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt, max_len=50):
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids]).to(config.device)
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.model(input_tensor)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                input_ids.append(next_token)
                input_tensor = torch.tensor([input_ids]).to(config.device)
                if next_token == self.tokenizer.token_to_id("[EOS]"):
                    break
        return self.tokenizer.decode(input_ids)

def summarize_text(text, tokenizer, model, max_len=50):
    tokens = tokenizer.encode(text)[:config.max_seq_len]
    input_ids = torch.tensor([tokens]).to(config.device)
    model.eval()
    summary_ids = []
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            summary_ids.append(next_token)
            input_ids = torch.tensor([tokens + summary_ids]).to(config.device)
            if next_token == tokenizer.token_to_id("[EOS]"):
                break
    return tokenizer.decode(summary_ids)

# 7. اجرای اصلی
if __name__ == "__main__":
    tokenizer = PersianTokenizer()
    tokenizer.train()

    dataset = PersianDataset(tokenizer)

    model = DariushGPT(config, tokenizer)
    trainer = DariushTrainer(model, tokenizer)
    trainer.train(dataset)

    qa = QAModel()
    print(qa.answer("ایران کشوری در خاورمیانه است.", "ایران کجاست؟"))

    poet = PoetryGenerator(model, tokenizer)
    print(poet.generate("شعر نو:"))

    print(summarize_text("متن طولانی درباره ایران...", tokenizer, model))
# Copyright (c) 2025 hosein davod abadi farahani

