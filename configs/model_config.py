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
    batch_size: int = 16  # Smaller batch size for more updates
    learning_rate: float = 1e-5  # Even smaller learning rate for stability
    num_epochs: int = 30  # More epochs
    warmup_ratio: float = 0.2  # Longer warmup
    weight_decay: float = 0.1  # Stronger regularization
    gradient_clip_val: float = 0.5  # Tighter gradient clipping
    
    # Data parameters
    data_dir: str = "data/skempi"
    train_val_split: float = 0.2
    random_seed: int = 42
    
    # Task-specific parameters
    regression_target: str = "delta_affinity"  # Target variable for prediction
    metrics: list = None  # Will be set to ["mae", "rmse", "pearson_corr"]
    
    def __post_init__(self):
        self.metrics = ["mae", "rmse", "pearson_corr"]
