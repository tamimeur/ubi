import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.regression import PearsonCorrCoef
import torch.nn.functional as F
from typing import Dict, Any

class AntibodyAffinityPredictor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Load ESM-2 model
        self.esm2 = AutoModel.from_pretrained(config.model_name)
        
        # Freeze most of ESM-2 layers, only fine-tune the last few
        for param in self.esm2.parameters():
            param.requires_grad = False
            
        # Unfreeze last N transformer layers
        for layer in self.esm2.encoder.layer[-2:]:  # Unfreeze last 2 layers
            for param in layer.parameters():
                param.requires_grad = True
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        # Metrics
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_pearson = PearsonCorrCoef()
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get ESM-2 embeddings
        outputs = self.esm2(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Pass through regression head
        prediction = self.regression_head(cls_embedding)
        return prediction.squeeze()
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        predictions = self(input_ids, attention_mask)
        loss = F.mse_loss(predictions, labels)
        
        # Log metrics
        self.train_mae(predictions, labels)
        self.log('train_loss', loss)
        self.log('train_mae', self.train_mae, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        predictions = self(input_ids, attention_mask)
        val_loss = F.mse_loss(predictions, labels)
        
        # Log metrics
        self.val_mae(predictions, labels)
        self.val_rmse(predictions, labels)
        self.val_pearson(predictions, labels)
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_mae', self.val_mae, prog_bar=True)
        self.log('val_rmse', self.val_rmse, prog_bar=True)
        self.log('val_pearson', self.val_pearson, prog_bar=True)
    
    def configure_optimizers(self):
        # Only optimize the unfrozen parameters
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.config.warmup_ratio
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
