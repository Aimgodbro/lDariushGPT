import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset
from pathlib import Path
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
            "[CLS]": 8, "[SEP]": 9
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

# 2. توکنایزر پیشرفته با پشتیبانی از Hugging Face Datasets
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
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
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

# 3. دیتاست هوشمند چندمنظوره
class PersianMultiTaskDataset(Dataset):
    def __init__(self, tokenizer, text_source="hf", poetry_source=None, sentiment_source=None):
        self.tokenizer = tokenizer
        self.data = []
        
        # بارگیری داده‌های متنی
        if text_source == "hf":
            dataset = load_dataset("oscar", "unshuffled_deduplicated_fa")
            self.texts = dataset["train"]["text"][:10000]
        
        # بارگیری داده‌های شعر
        if poetry_source:
            self.poems = self._load_poetry_data(poetry_source)
        
        # بارگیری داده‌های احساسات
        if sentiment_source:
            self.sentiments = self._load_sentiment_data(sentiment_source)

    def _load_poetry_data(self, path):
        return [{"text": "به نام خداوند جان و خرد", "bahr": "hazaj", "rhyme": "ar"}]

    def _load_sentiment_data(self, path):
        return [("این فیلم عالی بود!", 2), ("خیلی بد بود!", 0)]

    def __len__(self):
        return 10000  # برای ساده‌سازی

    def __getitem__(self, idx):
        task_type = random.choice(["text", "poetry", "sentiment"])
        
        if task_type == "text":
            text = self.texts[idx % len(self.texts)]
            tokens = self.tokenizer.encode(text)
            chunk = tokens[:config.max_seq_len]
            return {"input_ids": chunk[:-1], "labels": chunk[1:], "task": "text"}
        
        elif task_type == "poetry":
            poem = self.poems[idx % len(self.poems)]
            tokens = self.tokenizer.encode(
                f"[BAHR]{poem['bahr']} [RHYME]{poem['rhyme']} {poem['text']}"
            )
            chunk = tokens[:config.max_seq_len]
            return {"input_ids": chunk[:-1], "labels": chunk[1:], "task": "poetry"}
        
        else:
            text, label = self.sentiments[idx % len(self.sentiments)]
            tokens = self.tokenizer.encode(f"[CLS] {text} [SEP]")
            return {"input_ids": tokens, "labels": label, "task": "sentiment"}

# 4. معماری پیشرفته ترانسفورمر
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
        
        # هدهای تخصصی
        self.ln_f = nn.LayerNorm(config.emb_size)
        self.text_head = nn.Linear(config.emb_size, config.vocab_size)
        self.sentiment_head = nn.Linear(config.emb_size, 3)
        
        # مقداردهی اولیه
        self.apply(self._init_weights)
        nn.init.normal_(self.pos_embed, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, task="text"):
        B, T = x.size()
        
        # محاسبات پایه
        x = self.embedding(x) + self.pos_embed[:, :T]
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        
        # انتخاب هد
        if task == "sentiment":
            return self.sentiment_head(x[:, 0])
        else:
            return self.text_head(self.ln_f(x))
    
    def generate(self, prompt, max_length=100, temperature=0.7, top_k=50):
        self.eval()
        generated = prompt.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                inputs = torch.tensor([generated[-self.config.max_seq_len:]], 
                                    device=self.config.device)
                logits = self(inputs)[0, -1]
                
                # نمونه‌گیری کنترل شده
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

# 5. سیستم آموزش پیشرفته
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
                labels = batch["labels"].to(config.device)
                task = batch["task"][0]
                
                self.optimizer.zero_grad()
                
                with autocast(enabled=config.use_amp):
                    outputs = self.model(inputs, task)
                    
                    if task == "sentiment":
                        loss = F.cross_entropy(outputs, labels)
                    else:
                        loss = F.cross_entropy(outputs.view(-1, config.vocab_size), 
                                             labels.view(-1))
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                if config.device.type == 'mps':
                    torch.mps.empty_cache()
            
            avg_loss = total_loss / len(loader)
            print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
            self._save_best_model(avg_loss)
            self._generate_samples()

    def _save_best_model(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            torch.save({
                "model_state": self.model.state_dict(),
                "tokenizer": self.tokenizer,
                "config": vars(config)
            }, "best_dariush_model.pt")
            print("✅ بهترین مدل ذخیره شد!")

    def _generate_samples(self):
        # تولید نمونه متن
        text = self.model.generate([self.tokenizer.special_tokens["[BOS]"]])
        print("\nمتن تولیدی:", self.tokenizer.decode(text))
        
        # تولید شعر
        poem = self.model.generate_poem()
        print("\nشعر تولیدی:", self.tokenizer.decode(poem))
        
        # تحلیل احساسات
        sentiment = self.model.analyze_sentiment("این یک شاهکار است!")
        print("\nتحلیل احساسات:", sentiment)

# اجرای اصلی
if __name__ == "__main__":
    try:
        # آماده‌سازی توکنایزر
        tokenizer = PersianTokenizer()
        tokenizer.train_from_hf()
        config.vocab_size = tokenizer.get_vocab_size()
        
        # آماده‌سازی دیتاست
        dataset = PersianMultiTaskDataset(tokenizer)
        
        # ایجاد و آموزش مدل
        model = DariushGPT(config, tokenizer)
        trainer = DariushTrainer(model, tokenizer)
        trainer.train(dataset)
        
    except Exception as e:
        print(f"❌ خطای بحرانی: {e}")
        exit(1)
