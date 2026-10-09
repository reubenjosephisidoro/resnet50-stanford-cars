"""
The training loop for Stanford Cars - ResNet50 fine-tuning with progressive unfreezing.
 
This script allows the dataset, model, and training to coordinate with each other. 
It supports three stages of progressive unfreezing:
- Stage 1: Train only fc, layer3, layer4
- Stage 2: Unfreeze layer2, keep earlier layers frozen
- Stage 3: Unfreeze layer1 and stem (conv1, bn1)
 
Each stage loads the best weights from the previous stage, allowing manual
progression and inspection between runs.
 
Likely Commands for Usage:
    python train.py --stage 1 --data-root /path/to/car_data
    python train.py --stage 2 --data-root /path/to/car_data
    python train.py --stage 3 --data-root /path/to/car_data
"""


from __future__ import annotations
from dataclasses import dataclass

import torch


# 1. CONFIGURATIONS

@dataclass
class TrainingConfig:
    """Global training settings."""

    data_root:str
    checkpoint_dir:str = "./checkpoints"
    device:str|None = None
    seed:int = 42

    # Dataset config
    image_size:int = 448
    batch_size:int = 32
    num_workers:int = 2
    val_split:float = 0.0

    # Model config
    num_classes:int = 196

    # Checkpoint loading
    load_checkpoint:str|None = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class StageConfig:
    """Per-stage hyperparameters"""

    stage:int
    epochs:int
    learning_rates:dict[str, float]  # {layer_name: lr}
    momentum:float = 0.9
    weight_decay:float = 1e-4
    label_smoothing:float = 0.1
    early_stopping_patience:int = 5


# Below are stage-specific configurations
STAGE_CONFIGS = {
    1: StageConfig(stage=1,
                   epochs=12,
                   learning_rates={"fc": 0.01,
                                   "layer4": 0.01,
                                   "layer3": 0.01}
                    ),
    2: StageConfig(stage=2,
                   epochs=7,
                   learning_rates={"fc": 0.001,
                                   "layer4": 2e-4,
                                   "layer3": 1e-4,
                                   "layer2": 5e-5}
                    ),
    3: StageConfig(stage=3,
                   epochs=10,
                   learning_rates={"fc": 1e-4,
                                   "layer4": 7e-5,
                                   "layer3": 5e-5,
                                   "layer2": 2e-5,
                                   "layer1": 5e-6,
                                   "conv1": 5e-6,
                                   "bn1": 5e-6}
                    ),
                }