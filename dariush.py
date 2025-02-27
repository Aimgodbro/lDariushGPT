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
from xformers.ops import memory_efficient_attention, sparse_attention
import mlflow
from datetime import datetime
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastapi import FastAPI, UploadFile
from PIL import Image
import onnxruntime as ort
from modAL import ActiveLearner
from bert_score import score
from bitsandbytes import quantize_4bit
import coremltools as ct

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
        self.top_k = 2
        self.batch_size = 32
        self.learning_rate = 2e-5
        self.num_epochs = 15
        self.dropout = 0.1
        self.device = self._detect_device()
        self.use_amp = True
        self.use_sparse = True
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

# 2. توکنایزر فارسی پیشرفته
class PersianTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE())
        self.special_tokens = config.special_tokens
        
    def train_from_hf(self, dataset_name="oscar", dataset_config="unshuffled_deduplicated_fa"):
        try:
            dataset = load_dataset(dataset_name, dataset_config)
            text_iterator = (text for text in dataset["train"]["text"])
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
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

# 3. دیتاست با Augmentation
class PersianMultiTaskDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.texts = load_dataset("oscar", "unshuffled_deduplicated_fa")["train"]["text"][:10000]
        self.task_distribution = [0.6, 0.3, 0.1]
        self.augmenter = AdvancedAugmenter()

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        task_type = random.choices(["text", "poetry", "sentiment"], weights=self.task_distribution, k=1)[0]
        text = self._augment_text(self.texts[idx % len(self.texts)])
        
        if task_type == "text":
            return self._process_text(text)
        elif task_type == "poetry":
            return self._process_poetry(text)
        else:
            return self._process_sentiment(text)

    def _augment_text(self, text):
        return self.augmenter.augment(text, methods=["synonym", "back_translation"])

    # بقیه متدها مانند قبل...

# 4. معماری پیشرفته با بهبودها
class MoE(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_size, num_experts)
        self.top_k = top_k
        
    def forward(self, x):
        gates = torch.softmax(self.gate(x), dim=-1)
        top_gates, top_indices = torch.topk(gates, k=self.top_k, dim=-1)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=2)
        return torch.einsum('bkse,bkseh->bsh', top_gates, expert_outputs[top_indices])

class EnhancedTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb_size = config.emb_size
        self.num_heads = config.num_heads
        self.head_dim = self.emb_size // self.num_heads
        self.use_sparse = config.use_sparse
        
        # لایه‌های پروجکشن
        self.q_proj = nn.Linear(self.emb_size, self.emb_size)
        self.k_proj = nn.Linear(self.emb_size, self.emb_size)
        self.v_proj = nn.Linear(self.emb_size, self.emb_size)
        
        # سیستم MoE پیشرفته
        self.moe = MoE(config.hidden_size, config.num_experts, config.top_k)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        self.ffn = nn.Linear(self.emb_size, self.emb_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, attention_mask=None):
        B, S, _ = x.shape
        rotary_emb = self.rotary_emb(x, seq_len=S)
        
        # اعمال Rotary Embedding
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim)
        q = self.rotary_emb.apply_rotary(q, rotary_emb)
        k = self.rotary_emb.apply_rotary(k, rotary_emb)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim)
        
        # Sparse Attention
        if self.use_sparse:
            attn_pattern = sparse_attention.SparseCS(
                pattern="axial", 
                layout="hstl_l"
            )
            attn_output = memory_efficient_attention(
                q, k, v, 
                attn_bias=attn_pattern
            )
        else:
            attn_output = memory_efficient_attention(q, k, v)
            
        attn_output = attn_output.view(B, S, self.emb_size)
        moe_output = self.moe(attn_output)
        return self.dropout(self.ffn(moe_output)) + x

# 5. سیستم آموزش با MLOps
class DariushTrainer:
    def __init__(self, model, tokenizer):
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.scaler = GradScaler(enabled=config.use_amp)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.evaluator = DariushEvaluator()
        mlflow.start_run()
        
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
                    loss = self._compute_loss(outputs, labels, task)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                mlflow.log_metric("batch_loss", loss.item())
            
            avg_loss = total_loss / len(loader)
            self._log_metrics(epoch, avg_loss)
            self._save_best_model(avg_loss)
            self._generate_samples()
            
    def _compute_loss(self, outputs, labels, task):
        if task == "sentiment":
            return F.cross_entropy(outputs, labels)
        return F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))

    def _log_metrics(self, epoch, avg_loss):
        mlflow.log_metrics({
            "epoch_loss": avg_loss,
            "perplexity": torch.exp(torch.tensor(avg_loss)).item()
        }, step=epoch)
        
    def export_onnx(self, path="dariush.onnx"):
        dummy_input = torch.randint(0, config.vocab_size, (1, config.max_seq_len)).to(config.device)
        torch.onnx.export(
            self.model, dummy_input, path,
            opset_version=17,
            input_names=['input_ids'],
            output_names=['logits']
        )
        mlflow.log_artifact(path)

# 6. اجرای اصلی با بهبودها
if __name__ == "__main__":
    try:
        # تنظیمات لاگ
        logging.basicConfig(
            filename='dariush.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # آموزش مدل
        tokenizer = PersianTokenizer()
        tokenizer.train_from_hf()
        config.vocab_size = tokenizer.get_vocab_size()
        
        dataset = PersianMultiTaskDataset(tokenizer)
        model = DariushGPT(config, tokenizer)
        trainer = DariushTrainer(model, tokenizer)
        trainer.train(dataset)
        
        # Export مدل
        trainer.export_onnx()
        
    except Exception as e:
        logging.error(f"خطای اصلی: {str(e)}", exc_info=True)
        raise
    finally:
        mlflow.end_run()
# Copyright (c) 2025 hosein davod abadi farahani

