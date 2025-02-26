import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset
from tqdm import tqdm
import random

# 1. پیکربندی پیشرفته
class DariushConfig:
    def __init__(self):
        # پارامترهای مدل
        self.vocab_size = 30000
        self.emb_size = 512
        self.num_heads = 8
        self.num_layers = 6
        self.hidden_size = 2048
        self.max_seq_len = 512
        
        # تنظیمات آموزش
        self.batch_size = 16
        self.learning_rate = 3e-4
        self.num_epochs = 10
        self.dropout = 0.1
        
        # بهینه‌سازی سخت‌افزار
        self.device = self._detect_device()
        self.use_amp = True
        self.gradient_checkpointing = True
        
        # توکن‌های ویژه
        self.special_tokens = {
            "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
            "[ME]": 4, "[ENDLINE]": 5, "[RHYME]": 6, "[BAHR]": 7,
            "[CLS]": 8, "[SEP]": 9, "[COT]": 10, "[TRANS]": 11
        }
        
    def _detect_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            torch.mps.set_per_process_memory_fraction(0.85)
            return torch.device("mps")
        else:
            return torch.device("cpu")

config = DariushConfig()

# 2. توکنایزر پیشرفته
class PersianTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE())
        self.special_tokens = config.special_tokens
        
    def train_from_hf(self, dataset_name="oscar", dataset_config="unshuffled_deduplicated_fa"):
        try:
            dataset = load_dataset(dataset_name, dataset_config)
            text_iterator = (text for text in dataset["train"]["text"])
            self._train(text_iterator)
        except Exception as e:
            raise RuntimeError(f"خطا در بارگیری دیتاست: {e}")

    def _train(self, text_iterator):
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

# 3. دیتاست چندمنظوره (بدون تغییر)
class PersianMultiTaskDataset(Dataset):
    # ... (همانند کد قبلی بدون تغییر)

# 4. معماری پیشرفته با قابلیت‌های جدید
class DariushGPT(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # لایه‌های اصلی
        self.embedding = nn.Embedding(config.vocab_size, config.emb_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.emb_size))
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.emb_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(config.num_layers)
        ])
        
        # هدهای تخصصی جدید
        self.ln_f = nn.LayerNorm(config.emb_size)
        self.text_head = nn.Linear(config.emb_size, config.vocab_size)
        self.sentiment_head = nn.Linear(config.emb_size, 3)
        self.translation_head = nn.Linear(config.emb_size, config.vocab_size)  # ترجمه
        self.cot_head = nn.Linear(config.emb_size, config.vocab_size)  # Chain-of-Thought
        
        # مقداردهی اولیه
        self.apply(self._init_weights)
        nn.init.normal_(self.pos_embed, std=1.0 / (self.config.emb_size ** 0.5))

    def _init_weights(self, module):
        # ... (همانند کد قبلی)

    def forward(self, x, attention_mask=None, task="text"):
        B, T = x.size()
        
        # محاسبات پایه
        x = self.embedding(x) + self.pos_embed[:, :T]
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask)
            else:
                x = layer(x, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        
        # انتخاب هد بر اساس تسک
        if task == "sentiment":
            return self.sentiment_head(x[:, 0])
        elif task == "translation":
            return self.translation_head(self.ln_f(x))
        elif task == "cot":
            return self.cot_head(self.ln_f(x))
        else:
            return self.text_head(self.ln_f(x))
    
    # 5. تولید متن با قابلیت Chain-of-Thought
    def generate_with_cot(self, prompt, max_length=100, temperature=0.7):
        self.eval()
        generated = prompt.copy()
        cot_flag = False
        
        with torch.no_grad():
            for _ in range(max_length):
                inputs = torch.tensor([generated[-self.config.max_seq_len:]], 
                                    device=self.config.device)
                attention_mask = torch.ones_like(inputs, device=self.config.device)
                
                # تشخیص خودکار حالت CoT
                if self.tokenizer.special_tokens["[COT]"] in generated:
                    cot_flag = True
                    logits = self(inputs, attention_mask=attention_mask, task="cot")[0, -1]
                else:
                    logits = self(inputs, attention_mask=attention_mask)[0, -1]
                
                # نمونه‌گیری
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated.append(next_token.item())
                if next_token.item() == self.tokenizer.special_tokens["[EOS]"]:
                    break
        return generated
    
    # 6. ترجمه خودکار
    def translate(self, text, max_length=100, temperature=0.7):
        tokens = self.tokenizer.encode(f"[TRANS] {text}").ids
        inputs = torch.tensor([tokens], device=self.config.device)
        translated = []
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self(inputs, task="translation")[0, -1]
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == self.tokenizer.special_tokens["[EOS]"]:
                    break
                translated.append(next_token.item())
                inputs = torch.cat([inputs, next_token.unsqueeze(0)], dim=1)
        
        return self.tokenizer.decode(translated)

# 7. سیستم آموزش پیشرفته با قابلیت‌های جدید
class DariushTrainer:
    # ... (همانند کد قبلی با افزودن تسک‌های جدید در آموزش)

# اجرای اصلی
if __name__ == "__main__":
    try:
        tokenizer = PersianTokenizer()
        tokenizer.train_from_hf()
        config.vocab_size = tokenizer.get_vocab_size()
        
        dataset = PersianMultiTaskDataset(tokenizer)
        model = DariushGPT(config, tokenizer)
        trainer = DariushTrainer(model, tokenizer)
        trainer.train(dataset)
        
    except Exception as e:
        print(f"❌ خطای بحرانی: {e}")
        exit(1)
# Copyright (c) 2025 hosein davod abadi farahani
# Licensed under the MIT License (https://opensource.org/licenses/MIT)
