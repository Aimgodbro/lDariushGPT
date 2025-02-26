import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset
from tqdm import tqdm
import random
import faiss
import numpy as np
from xformers.ops import memory_efficient_attention

# 1. پیکربندی پیشرفته
class DariushConfig:
    def __init__(self):
        self.vocab_size = 50000
        self.emb_size = 1024
        self.num_heads = 16
        self.num_layers = 12
        self.hidden_size = 4096
        self.max_seq_len = 2048
        self.num_experts = 8
        self.batch_size = 32
        self.learning_rate = 2e-5
        self.num_epochs = 15
        self.dropout = 0.1
        self.device = self._detect_device()
        self.use_amp = True
        self.special_tokens = {
            "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
            "[ME]": 4, "[ENDLINE]": 5, "[RHYME]": 6, "[BAHR]": 7,
            "[CLS]": 8, "[SEP]": 9, "[COT]": 10, "[TRANS]": 11,
            "[IMG]": 12, "[AUD]": 13, "[MEM]": 14
        }
        
    def _detect_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

config = DariushConfig()

# 2. توکنایزر فارسی بهبود یافته
class PersianTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE())
        self.special_tokens = config.special_tokens
        
    def train_from_hf(self, dataset_name="oscar", dataset_config="unshuffled_deduplicated_fa"):
        dataset = load_dataset(dataset_name, dataset_config)
        text_iterator = (text for text in dataset["train"]["text"])
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=config.vocab_size,
            special_tokens=list(self.special_tokens.keys()),
            min_frequency=2
        )
        self.tokenizer.train_from_iterator(text_iterator, trainer=trainer)
        
    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text).ids
        
    def decode(self, tokens: list) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

# 3. دیتاست چندمنظوره اصلاح شده
class PersianMultiTaskDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.texts = load_dataset("oscar", "unshuffled_deduplicated_fa")["train"]["text"][:10000]
        self.task_distribution = [0.6, 0.3, 0.1]

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        task_type = random.choices(["text", "poetry", "sentiment"], weights=self.task_distribution, k=1)[0]
        
        if task_type == "text":
            return self._process_text(idx)
        elif task_type == "poetry":
            return self._process_poetry(idx)
        else:
            return self._process_sentiment(idx)

    def _pad_sequence(self, tokens, max_len, pad_token):
        padded = tokens[:max_len] + [pad_token] * (max_len - len(tokens))
        mask = [1] * min(len(tokens), max_len) + [0] * (max_len - len(tokens))
        return padded, mask

    def _process_text(self, idx):
        text = self.texts[idx % len(self.texts)]
        tokens = self.tokenizer.encode(text).ids[:config.max_seq_len-1]
        tokens += [self.tokenizer.special_tokens["[EOS]"]]
        input_ids, attention_mask = self._pad_sequence(tokens, config.max_seq_len, 
                                                     self.tokenizer.special_tokens["[PAD]"])
        return {
            "input_ids": torch.tensor(input_ids[:-1]),
            "labels": torch.tensor(input_ids[1:]),
            "attention_mask": torch.tensor(attention_mask[:-1]),
            "task": "text"
        }

    def _process_poetry(self, idx):
        poem = {
            "text": "به نام خداوند جان و خرد",
            "bahr": "hazaj",
            "rhyme": "ar"
        }
        tokens = self.tokenizer.encode(f"[BAHR]{poem['bahr']} [RHYME]{poem['rhyme']} {poem['text']}").ids
        input_ids, attention_mask = self._pad_sequence(tokens, config.max_seq_len, 
                                                     self.tokenizer.special_tokens["[PAD]"])
        return {
            "input_ids": torch.tensor(input_ids[:-1]),
            "labels": torch.tensor(input_ids[1:]),
            "attention_mask": torch.tensor(attention_mask[:-1]),
            "task": "poetry"
        }

    def _process_sentiment(self, idx):
        cases = [
            ("خیلی بد بود!", 0), ("بدک نبود", 1), ("عالی بود!", 2)
        ]
        text, label = random.choice(cases)
        tokens = self.tokenizer.encode(f"[CLS] {text} [SEP]").ids
        input_ids, attention_mask = self._pad_sequence(tokens, config.max_seq_len,
                                                     self.tokenizer.special_tokens["[PAD]"])
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(label),
            "attention_mask": torch.tensor(attention_mask),
            "task": "sentiment"
        }

# 4. معماری پیشرفته با اصلاحات
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        
    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, None, :]
    
    def apply_rotary(self, x, emb):
        cos = emb.cos()
        sin = emb.sin()
        return (x * cos) + (self.rotate_half(x) * sin)

class MoE(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_size, num_experts)
        
    def forward(self, x):
        gates = torch.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=2)
        return torch.einsum('bse,bseh->bsh', gates, expert_outputs)

class EnhancedTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb_size = config.emb_size
        self.num_heads = config.num_heads
        self.head_dim = self.emb_size // self.num_heads
        
        self.q_proj = nn.Linear(self.emb_size, self.emb_size)
        self.k_proj = nn.Linear(self.emb_size, self.emb_size)
        self.v_proj = nn.Linear(self.emb_size, self.emb_size)
        self.moe = MoE(config.hidden_size, config.num_experts)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        self.ffn = nn.Linear(self.emb_size, self.emb_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, attention_mask=None):
        B, S, _ = x.shape
        rotary_emb = self.rotary_emb(x, seq_len=S)
        
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim)
        q = self.rotary_emb.apply_rotary(q, rotary_emb)
        k = self.rotary_emb.apply_rotary(k, rotary_emb)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim)
        
        attn_output = memory_efficient_attention(
            q, k, v, attn_bias=attention_mask
        )
        attn_output = attn_output.view(B, S, self.emb_size)
        moe_output = self.moe(attn_output)
        return self.dropout(self.ffn(moe_output)) + x

class DariushGPT(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        self.embedding = nn.Embedding(config.vocab_size, config.emb_size)
        self.layers = nn.ModuleList([EnhancedTransformerLayer(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.emb_size)
        self.text_head = nn.Linear(config.emb_size, config.vocab_size)
        self.sentiment_head = nn.Linear(config.emb_size, 3)
        
        # سیستم حافظه
        self.retriever = faiss.IndexFlatL2(config.emb_size)
        self.memory = {}
        self.image_proj = nn.Linear(512, config.emb_size)
        self.audio_proj = nn.Linear(256, config.emb_size)

    def forward(self, x, attention_mask=None, task="text"):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.ln_f(x)
        
        if task == "sentiment":
            return self.sentiment_head(x[:, 0])
        return self.text_head(x)
    
    def generate(self, prompt, max_length=100, temperature=0.7, top_k=50):
        self.eval()
        generated = prompt.copy()
        with torch.no_grad():
            for _ in range(max_length):
                inputs = torch.tensor([generated[-self.config.max_seq_len:]], device=self.config.device)
                logits = self(inputs)[0, -1]
                logits = logits / temperature
                top_values, top_indices = torch.topk(logits, top_k)
                probs = F.softmax(top_values, dim=-1)
                next_token = top_indices[torch.multinomial(probs, num_samples=1)]
                generated.append(next_token.item())
                if next_token.item() == self.tokenizer.special_tokens["[EOS]"]:
                    break
        return generated
    
    def generate_poem(self, bahr="hazaj", rhyme="ar"):
        prompt = [
            self.tokenizer.special_tokens["[BAHR]"],
            self.tokenizer.special_tokens["[RHYME]"],
            self.tokenizer.special_tokens["[BOS]"]
        ]
        return self.generate(prompt, temperature=0.8, max_length=100)
    
    def analyze_sentiment(self, text):
        tokens = self.tokenizer.encode(f"[CLS] {text} [SEP]")
        inputs = torch.tensor([tokens], device=self.config.device)
        logits = self(inputs, task="sentiment")
        return ["منفی", "خنثی", "مثبت"][logits.argmax().item()]

# 5. سیستم آموزش کامل شده
class DariushTrainer:
    def __init__(self, model, tokenizer):
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.scaler = GradScaler(enabled=config.use_amp)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.best_loss = float('inf')

    def train(self, dataset):
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        for epoch in range(config.num_epochs):
            self.model.train()
            total_loss = 0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
            for batch in pbar:
                inputs = batch["input_ids"].to(config.device)
                masks = batch["attention_mask"].to(config.device)
                labels = batch["labels"].to(config.device)
                task = batch["task"][0]
                
                self.optimizer.zero_grad()
                with autocast(enabled=config.use_amp):
                    outputs = self.model(inputs, attention_mask=masks, task=task)
                    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1)) if task != "sentiment" else F.cross_entropy(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
            self._save_best_model(avg_loss)
            self._generate_samples()

    def _save_best_model(self, current_loss):
        if current_loss < self.best_loss:
            torch.save(self.model.state_dict(), "best_dariush_model.pt")
            print("مدل بهبود یافته ذخیره شد!")

    def _generate_samples(self):
        # نمونه‌های تولیدی
        text = self.model.generate([self.tokenizer.special_tokens["[BOS]"]])
        print("متن تولیدی:", self.tokenizer.decode(text))
        
        poem = self.model.generate_poem()
        print("شعر تولیدی:", self.tokenizer.decode(poem))
        
        sentiment = self.model.analyze_sentiment("این یک شاهکار است!")
        print("تحلیل احساسات:", sentiment)

# اجرای اصلی
if __name__ == "__main__":
    tokenizer = PersianTokenizer()
    tokenizer.train_from_hf()
    config.vocab_size = tokenizer.get_vocab_size()
    
    dataset = PersianMultiTaskDataset(tokenizer)
    model = DariushGPT(config, tokenizer)
    trainer = DariushTrainer(model, tokenizer)
    trainer.train(dataset)
# Copyright (c) 2025 hosein davod abadi farahani
# Licensed under the MIT License (https://opensource.org/licenses/MIT)
