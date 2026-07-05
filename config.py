from dataclasses import dataclass
import torch

@dataclass
class Config:
    num_points: int = 1024
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    num_classes: int = 40
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ablation and architectural flags
    use_input_transform: bool = True
    use_feature_transform: bool = True
    pooling_type: str = "max"          # "max" | "avg"
    reg_weight: float = 0.001          # scale for the feature transform regularization
    use_augmentation: bool = True
