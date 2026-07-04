from dataclasses import dataclass
import torch

@dataclass
class Config:
    num_points: int = 1024
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 250
    num_classes: int = 40
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
