"""The data pipeline for the Stanford Cars ResNet50 fine-tuning experiment.
This has been extracted from the exploratory notebook, covering:

    - The discovery (and retrieval) of image paths and their 
      string labels from a class-folder layout
    - The building of a canonical class-name->integer-id mapping
    - Resize + pad transforms that ensures to preserve aspect ratio
    - The CarImageDataset map-style dataset
    - Construction of a deterministic DataLoader object

This module can be run and exercised on its own because it does not import 
the model or the training loop: python dataset.py --data-root /path/to/car_data
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

__all__ = [
    "DataConfig",
    "CarImageDataset",
    "ResizeLongestSide",
    "PadToSquare",  
    "get_transforms",
    "get_img_paths_and_labels",
    "load_class_names",
    "build_label_map",
    "build_datasets",
    "build_dataloaders",
    "set_seed",
    "seed_worker"]

# ImageNet statistics, since the backbone is pretrained on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 196


# 1. CONFIGURATION
@dataclass
class DataConfig:
    """Everything the data pipeline needs.
    train.py builds one of these and passes it to build_dataloaders(),
    so image size/normalization constants are all here in one place.
    """

    train_dir: str
    test_dir: str
    cars_meta_path: str | None = None
    image_size: int = 448
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 42

    # Fraction of the training set that will be taken out for validation. 
    # The updated notebook ran with 0.0 (evaluating directly on the test set) 
    # Keep that as the default so results stay comparable, 
    # but make a real holdout one argument away.
    val_split: float = 0.0
    mean: Sequence[float] = IMAGENET_MEAN
    std: Sequence[float] = IMAGENET_STD


# 2. REPRODUCIBILITY
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs in the pipeline.
    Note: does not affect DataLoader workers, 
    which are handled separately in seed_worker().
    """

    # Seed for the main processes
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For cuDNN determinism
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - signature fixed by torch
    """Give each DataLoader worker a deterministic but 
    distinct seed when they are instantiated."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# 3. AUGMENTATIONS
class ResizeLongestSide:
    """For resizing the image's longest side to a fixed size with 
    the goal of preserving aspect ratio.

    Here, it becomes a callable class rather than transforms.Lambda wrapping a
    closure. Lambdas cannot be pickled, which breaks `num_workers > 0` under
    the "spawn" start method and prevents checkpointing a transform object.
    """
    def __init__(self, size:int, resample=Image.LANCZOS):
        self.size = size
        self.resample = resample

    def __call__(self, image:Image.Image) -> Image.Image:
        width, height = image.size
        if width >= height:
            new_w, new_h = self.size, max(1, round(height * self.size / width))
        else:
            new_w, new_h = max(1, round(width * self.size / height)), self.size
        return image.resize((new_w, new_h), self.resample)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


class PadToSquare:
    """Pad the shorter side so the image 
    becomes centered with squared canvas."""

    def __init__(self, fill: int | tuple[int, int, int]=0):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        longest = max(width, height)
        pad_w = max(0, longest - width)
        pad_h = max(0, longest - height)
        border = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2)
        return ImageOps.expand(image, border=border, fill=self.fill)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fill={self.fill})"


def get_transforms(
    train = True,
    image_size = 448,
    mean = IMAGENET_MEAN,
    std = IMAGENET_STD) -> transforms.Compose:
    """
    Build the augmentation pipeline.
    Photometric augmentations go first then geometric ones.
    """
    geometric_augs = [ResizeLongestSide(image_size), PadToSquare()]

    if train:
        stages = (
            [transforms.ColorJitter(brightness=0.2, contrast=0.2)]
            + geometric_augs
            + [transforms.RandomHorizontalFlip()])
    else:
        stages = list(geometric_augs)

    stages += [
        transforms.ToTensor(),
        transforms.Normalize(list(mean), list(std))]
    
    return transforms.Compose(stages)   