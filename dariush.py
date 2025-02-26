"""
Dariush GPT - نسخه نهایی با Hydra و ساختار سازمانی
"""

import os
import hydra
import torch
import mlflow
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from xformers.ops import memory_efficient_attention, sparse_attention
from tokenizers import Tokenizer, models, trainers

# ماژول‌های داخلی
from src.data import PersianDatasetBuilder
from src.model import DariushGPT
from src.utils import PersianAugmenter, PoetryValidator

# تنظیمات پایه لاگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('dariush.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """تابع اصلی اجرای پروژه"""
    
    # 1. آماده‌سازی اولیه
    logger.info("شروع آموزش با پیکربندی:\n%s", OmegaConf.to_yaml(cfg))
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    
    # 2. آماده‌سازی داده‌ها
    data_builder = PersianDatasetBuilder(cfg.data)
    tokenizer = data_builder.build_tokenizer()
    train_dataset, val_dataset = data_builder.build_datasets()
    
    # 3. مدل و آموزش
    with mlflow.start_run():
        # لاگ پارامترها
        mlflow.log_params(OmegaConf.to_container(cfg.model))
        mlflow.log_params(OmegaConf.to_container(cfg.train))
        
        # مقداردهی اولیه مدل
        model = DariushGPT(
            vocab_size=tokenizer.get_vocab_size(),
            **cfg.model.architecture
        )
        trainer = AdvancedTrainer(
            model=model,
            cfg=cfg.train,
            tokenizer=tokenizer
        )
        
        # آموزش
        trainer.train(train_dataset, val_dataset)
        
        # ذخیره نهایی
        trainer.save_model()
        logger.info("آموزش با موفقیت به پایان رسید")

class AdvancedTrainer:
    """سیستم آموزش پیشرفته با قابلیت‌های سازمانی"""
    
    def __init__(self, model, cfg, tokenizer):
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = torch.device(cfg.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        
        # بهینه‌ساز
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        
    def train(self, train_dataset, val_dataset):
        """حلقه اصلی آموزش"""
        train_loader = self._build_dataloader(train_dataset)
        val_loader = self._build_dataloader(val_dataset)
        
        for epoch in range(self.cfg.epochs):
            # آموزش
            self._run_epoch(train_loader, is_training=True)
            
            # اعتبارسنجی
            val_metrics = self._run_epoch(val_loader, is_training=False)
            
            # لاگ متریک‌ها
            self._log_metrics(epoch, val_metrics)
            
            # ذخیره بهینه
            if val_metrics['loss'] < self.best_loss:
                self.save_model()
                self.best_loss = val_metrics['loss']

    def _run_epoch(self, loader, is_training):
        """اجرای یک دوره کامل"""
        self.model.train(is_training)
        total_loss = 0
        progress_bar = tqdm(loader, desc=f'{"آموزش" if is_training else "اعتبارسنجی"}', leave=False)
        
        with torch.set_grad_enabled(is_training):
            for batch in progress_bar:
                inputs = batch['input_ids'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                # محاسبات ترکیبی
                with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                    outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, targets)
                
                # بهینه‌سازی
                if is_training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
        
        return {'loss': total_loss / len(loader)}

    def _compute_loss(self, outputs, targets):
        """تابع محاسبه loss چندمنظوره"""
        return F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=self.tokenizer.token_to_id('[PAD]')
        )

    def save_model(self):
        """ذخیره مدل با قابلیت‌های پیشرفته"""
        save_path = Path(self.cfg.save_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path.mkdir(parents=True, exist_ok=True)
        
        # ذخیره همه اجزا
        torch.save(self.model.state_dict(), save_path / 'model.pt')
        self.tokenizer.save(str(save_path / 'tokenizer.json'))
        
        # Export به ONNX
        self._export_onnx(save_path)
        logger.info("مدل در %s ذخیره شد", save_path)
        
        # لاگ در MLflow
        mlflow.log_artifacts(save_path)

    def _export_onnx(self, path):
        """تبدیل مدل به فرمت ONNX"""
        dummy_input = torch.randint(
            low=0,
            high=self.tokenizer.get_vocab_size(),
            size=(1, self.cfg.seq_len),
            device=self.device
        )
        torch.onnx.export(
            self.model,
            dummy_input,
            path / 'model.onnx',
            opset_version=17,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={'input_ids': {0: 'batch', 1: 'seq_len'}}
        )

if __name__ == "__main__":
    # تنظیمات اولیه Hydra
    os.environ['HYDRA_FULL_ERROR'] = '1'
    main()
# Copyright (c) 2025 hosein davod abadi farahani

