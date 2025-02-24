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
        
        # Normalize labels
        labels_mean = labels.mean()
        labels_std = labels.std().clamp(min=1e-6)  # Avoid division by zero
        normalized_labels = (labels - labels_mean) / labels_std
        
        predictions = self(input_ids, attention_mask)
        
        # Check for NaN values
        if torch.isnan(predictions).any() or torch.isnan(normalized_labels).any():
            self.log('nan_detected', 1.0)
            return None
        
        # Use normalized labels for loss
        loss = F.mse_loss(predictions, normalized_labels)
        
        # Denormalize predictions for metric calculation
        denorm_predictions = predictions * labels_std + labels_mean
        
        # Log metrics
        self.train_mae(denorm_predictions, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_mae', self.train_mae, prog_bar=True)
        
        # Log additional metrics
        self.log('batch_labels_mean', labels_mean, on_step=True)
        self.log('batch_labels_std', labels_std, on_step=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Normalize labels
        labels_mean = labels.mean()
        labels_std = labels.std().clamp(min=1e-6)  # Avoid division by zero
        normalized_labels = (labels - labels_mean) / labels_std
        
        predictions = self(input_ids, attention_mask)
        val_loss = F.mse_loss(predictions, normalized_labels)
        
        # Denormalize predictions for metric calculation
        denorm_predictions = predictions * labels_std + labels_mean
        
        # Log metrics
        self.val_mae(denorm_predictions, labels)
        self.val_rmse(denorm_predictions, labels)
        self.val_pearson(denorm_predictions, labels)
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_mae', self.val_mae, prog_bar=True)
        self.log('val_rmse', self.val_rmse, prog_bar=True)
        self.log('val_pearson', self.val_pearson, prog_bar=True)
        
        # Log additional metrics
        self.log('val_labels_mean', labels_mean)
        self.log('val_labels_std', labels_std)
    
    def configure_optimizers(self):
        # Only optimize the unfrozen parameters
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-7  # Increased epsilon for better numerical stability
        )
        
        # Learning rate scheduler with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.config.warmup_ratio,
            div_factor=25.0,
            final_div_factor=1000.0,
            three_phase=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
