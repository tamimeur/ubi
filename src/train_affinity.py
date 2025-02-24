import os
import sys
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb

from data.dataset import AntibodyAffinityDataset
from models.affinity_model import AntibodyAffinityPredictor
from configs.model_config import AntibodyAffinityConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(config: AntibodyAffinityConfig):
    # Initialize wandb
    wandb_logger = WandbLogger(project='antibody-affinity', name='esm2-finetuning')
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = AntibodyAffinityDataset(config.data_dir, split='train')
    val_dataset = AntibodyAffinityDataset(config.data_dir, split='val')
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    logger.info("Initializing ESM2 model for fine-tuning...")
    model = AntibodyAffinityPredictor(config)
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='esm2-affinity-{epoch:02d}-{val_pearson:.3f}',
            monitor='val_pearson',
            mode='max',
            save_top_k=3
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=config.gradient_clip_val,
        deterministic=True
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Close wandb run
    wandb.finish()
    logger.info("Training completed!")

if __name__ == '__main__':
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Load config and start training
    config = AntibodyAffinityConfig()
    train(config)
