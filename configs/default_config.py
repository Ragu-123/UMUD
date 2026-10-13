from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class Config:
    # Paths (Default Kaggle paths)
    data_dir: str = "/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data"
    output_dir: str = "/kaggle/working/checkpoints"
    submission_path: str = "/kaggle/working/submission.csv"

    # Model & Architecture
    backbone: str = "resnet34"          # 'resnet34', 'efficientnet_b3', or 'standalone'
    pretrained: bool = True
    in_channels: int = 3
    num_classes: int = 1
    image_size: Tuple[int, int] = (512, 512)

    # Training Hyperparameters
    batch_size: int = 8
    num_workers: int = 2
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    epochs: int = 25
    num_folds: int = 5
    seed: int = 42

    # Metrology Parameters
    structure_tensor_sigma: float = 3.0
    pa_clip_min: float = 5.0
    pa_clip_max: float = 45.0
    fl_clip_min: float = 30.0
    fl_clip_max: float = 200.0
    mt_clip_min: float = 10.0
    mt_clip_max: float = 50.0

    # Official Metric Normalization Tolerances
    tau_pa: float = 6.0
    tau_fl: float = 12.0
    tau_mt: float = 3.0
