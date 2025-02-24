from dataclasses import dataclass
from typing import Optional

@dataclass
class AntibodyAffinityConfig:
    # Model parameters
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    max_length: int = 512
    hidden_size: int = 1280  # ESM2's hidden size
    num_hidden_layers: int = 2
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 2e-5  # Smaller learning rate for fine-tuning
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    
    # Data parameters
    data_dir: str = "data/skempi"
    train_val_split: float = 0.2
    random_seed: int = 42
    
    # Task-specific parameters
    regression_target: str = "delta_affinity"  # Target variable for prediction
    metrics: list = None  # Will be set to ["mae", "rmse", "pearson_corr"]
    
    def __post_init__(self):
        self.metrics = ["mae", "rmse", "pearson_corr"]
