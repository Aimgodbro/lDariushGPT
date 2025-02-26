import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from hydra import compose, initialize
from omegaconf import OmegaConf
import pytorch_lightning as pl
from tokenizers import Tokenizer
from datasets import load_dataset

# 1. پیکربندی چندسکویی با Hydra
class GlobalConfig(pl.LightningDataModule):
    def __init__(self, config_path="config", config_name="main"):
        super().__init__()
        with initialize(config_path=config_path, job_name="dariush_config"):
            self.cfg = compose(config_name=config_name)
        
        # تنظیم خودکار دستگاه بر اساس محیط
        self.device = self._auto_detect_device()
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
    def _auto_detect_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

# 2. معماری اصلی با پشتیبانی توزیع‌شده
class DariushGPT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.tokenizer = self._init_tokenizer()
        
        # معماری پایه
        self.model = nn.Transformer(
            d_model=config.emb_size,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.hidden_size
        )
        
        # هدهای تخصصی
        self.heads = nn.ModuleDict({
            'text': nn.Linear(config.emb_size, config.vocab_size),
            'sentiment': nn.Linear(config.emb_size, 3),
            'translation': nn.Linear(config.emb_size, config.vocab_size)
        })

    def _init_tokenizer(self):
        # ... (همانند کد قبلی بدون تغییر)

    def forward(self, src, tgt=None, task='text'):
        # ... (همانند کد قبلی بدون تغییر)

    def training_step(self, batch, batch_idx):
        # منطق آموزش توزیع‌شده
        inputs, labels, task = batch
        output = self(inputs, task=task)
        loss = F.cross_entropy(output, labels)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.config.lr)

# 3. سیستم داده توزیع‌شده
class PersianDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = PersianMultiTaskDataset()  # بدون تغییر
        
    def train_dataloader(self):
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank
        ) if self.trainer.world_size > 1 else None
        
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=os.cpu_count()
        )

# 4. رابط اجرای ترکیبی
def main():
    # تنظیمات خودکار بر اساس محیط
    config = GlobalConfig()
    
    # آموزش با PyTorch Lightning
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        strategy='ddp' if config.world_size > 1 else 'auto',
        precision='16-mixed' if config.device == 'cuda' else '32-true',
        max_epochs=config.cfg.epochs,
        enable_progress_bar=config.world_size == 1
    )
    
    model = DariushGPT(config.cfg)
    dm = PersianDataModule(config.cfg)
    
    trainer.fit(model, dm)

# 5. اسکریپت اجرا برای محیط‌های مختلف
if __name__ == "__main__":
    # تنظیمات پیش‌فرض
    os.environ["HYDRA_FULL_ERROR"] = "1"
    
    # اجرای توزیع‌شده روی ابررایانه
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        main()
        dist.destroy_process_group()
    else:
        # اجرای محلی
        main()
# Copyright (c) 2025 hosein davod abadi farahani
# Licensed under the MIT License (https://opensource.org/licenses/MIT)
