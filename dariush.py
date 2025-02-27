import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np
import mlflow
from datetime import datetime
import logging

# 1. پیکربندی پیشرفته با توضیحات
class DariushConfig:
    def __init__(self):
        self.vocab_size = 50000  # اندازه واژگان برای توکنایزر
        self.emb_size = 1024     # اندازه embedding برای هر توکن
        self.num_heads = 16      # تعداد هدهای توجه
        self.num_layers = 12     # تعداد لایه‌های ترانسفورمر
        self.hidden_size = 4096  # اندازه لایه مخفی در FeedForward
        self.max_seq_len = 2048  # حداکثر طول توالی ورودی
        self.num_experts = 8     # تعداد کارشناسان در MoE
        self.top_k = 2          # تعداد کارشناسان برتر انتخاب‌شده
        self.batch_size = self._dynamic_batch_size()  # تنظیم پویا
        self.learning_rate = 2e-5
        self.num_epochs = 15
        self.dropout = 0.1
        self.device = self._detect_device()
        self.use_amp = True      # Mixed Precision فعال
        self.use_sparse = True   # Sparse Attention فعال
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
        return torch.device("cpu")

    def _dynamic_batch_size(self):
        if self.device.type == "cuda":
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            return min(32, int(mem * 4))  # حداکثر 32 یا متناسب با حافظه
        return 16  # پیش‌فرض برای CPU/MPS

config = DariushConfig()

# 2. توکنایزر (بدون تغییر زیاد)
class PersianTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE())
        self.special_tokens = config.special_tokens
        
    def train_from_hf(self, dataset_name="oscar", dataset_config="unshuffled_deduplicated_fa"):
        try:
            dataset = load_dataset(dataset_name, dataset_config, split="train[:10%]")  # محدود به 10% برای سرعت
            text_iterator = (text for text in dataset["text"])
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            trainer = trainers.BpeTrainer(
                vocab_size=config.vocab_size,
                special_tokens=list(self.special_tokens.keys()),
                min_frequency=2
            )
            self.tokenizer.train_from_iterator(text_iterator, trainer=trainer)
            self.save(f"tokenizers/fa-bpe-{datetime.now().strftime('%Y%m%d')}")
        except Exception as e:
            logging.error(f"خطا در آموزش توکنایزر: {str(e)}")
            raise
        
    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text).ids
        
    def decode(self, tokens: list) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)
    
    def save(self, path):
        self.tokenizer.save(path)

# 3. دیتاست با پیاده‌سازی کامل
class PersianMultiTaskDataset(Dataset):
    def __init__(self, tokenizer, split="train"):
        self.tokenizer = tokenizer
        self.split = split
        self.texts = load_dataset("oscar", "unshuffled_deduplicated_fa", split=split)["text"][:10000]
        self.task_distribution = [0.6, 0.3, 0.1]  # احتمال وظایف مختلف
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        task_type = random.choices(["text", "poetry", "sentiment"], weights=self.task_distribution, k=1)[0]
        text = self.texts[idx % len(self.texts)]
        
        if task_type == "text":
            return self._process_text(text)
        elif task_type == "poetry":
            return self._process_poetry(text)
        else:
            return self._process_sentiment(text)

    def _process_text(self, text):
        tokens = self.tokenizer.encode(text)[:config.max_seq_len]
        input_ids = tokens[:-1]
        labels = tokens[1:]
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.ones_like(torch.tensor(input_ids)),
            "task": "text"
        }

    def _process_poetry(self, text):
        # فرض ساده: تکرار متن به‌عنوان برچسب
        tokens = self.tokenizer.encode(text + " [ENDLINE]")[:config.max_seq_len]
        input_ids = tokens[:-1]
        labels = tokens[1:]
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.ones_like(torch.tensor(input_ids)),
            "task": "poetry"
        }

    def _process_sentiment(self, text):
        tokens = self.tokenizer.encode(text)[:config.max_seq_len]
        # فرض: برچسب مثبت (1) یا منفی (0) به‌صورت تصادفی
        label = random.randint(0, 1)
        return {
            "input_ids": torch.tensor(tokens),
            "labels": torch.tensor(label),
            "attention_mask": torch.ones_like(torch.tensor(tokens)),
            "task": "sentiment"
        }

# 4. پیاده‌سازی Rotary Embedding (جدید)
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    def apply_rotary(self, x, rotary_emb):
        B, S, H, D = x.shape
        sin_val = torch.sin(rotary_emb).to(x.device)
        cos_val = torch.cos(rotary_emb).to(x.device)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x[..., 0::2] = x1 * cos_val - x2 * sin_val
        x[..., 1::2] = x1 * sin_val + x2 * cos_val
        return x

    def forward(self, x, seq_len):
        pos = torch.arange(seq_len, device=x.device).float()
        rotary_emb = pos[:, None] * self.inv_freq[None, :]
        return rotary_emb

# 5. اجرای اصلی
if __name__ == "__main__":
    logging.basicConfig(filename='dariush.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        tokenizer = PersianTokenizer()
        tokenizer.train_from_hf()
        config.vocab_size = tokenizer.tokenizer.get_vocab_size()
        
        train_dataset = PersianMultiTaskDataset(tokenizer, split="train[:80%]")
        val_dataset = PersianMultiTaskDataset(tokenizer, split="train[80%:90%]")
        
        # فرض: DariushGPT پیاده‌سازی شده باشد
        # model = DariushGPT(config, tokenizer)
        # trainer = DariushTrainer(model, tokenizer)
        # trainer.train(train_dataset)
        
    except Exception as e:
        logging.error(f"خطای اصلی: {str(e)}", exc_info=True)
        raise
    finally:
        mlflow.end_run()
# Copyright (c) 2025 hosein davod abadi farahani

