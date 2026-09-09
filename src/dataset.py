"""
The data pipeline for the Stanford Cars ResNet50 fine-tuning experiment.
This has been extracted from the exploratory notebook, covering:

    - The discovery and retrieval of image paths and their 
      string labels from a class-folder layout
    - The building of a canonical class name --> integer id mapping
    - Resize + pad transforms that ensures aspect ratio is preserved
    - The CarImageDataset map-style dataset
    - Construction of a DataLoader object with deterministic behavior

This module can be run on its own because it does not import 
the model or the training loop: python dataset.py --data-root /path/to/car_data
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable, Iterable, Sequence

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

    train_dir:str
    test_dir:str
    cars_meta_path:str|None = None
    image_size:int = 448
    batch_size:int = 32
    num_workers:int = 2
    pin_memory:bool = True
    seed:int = 42

    # Fraction of the training set that will be taken out for validation. 
    # The updated notebook ran with 0.0 val split (evaluating directly 
    # on the test set). Keep that as the default so results 
    # stay comparable, but make a validation holdout easily doable.
    val_split:float = 0.0
    mean:Sequence[float] = IMAGENET_MEAN
    std:Sequence[float] = IMAGENET_STD


# 2. REPRODUCIBILITY
def set_seed(seed=42, deterministic=True) -> None:
    """Seed all RNGs in the pipeline.
    Note: this does not affect DataLoader workers, 
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


def seed_worker(worker_id:int) -> None:  
    """Give each DataLoader worker a deterministic but 
    distinct seed when they are instantiated."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_generator(seed) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# 3. AUGMENTATIONS
class ResizeLongestSide:
    """For resizing the image's longest side to a fixed size with 
    the goal of preserving aspect ratio.

    Here, it becomes a callable class rather than transforms.Lambda wrapping a
    closure. Lambdas cannot be pickled, which breaks num_workers>0 under
    the "spawn" start method and prevents checkpointing a transform object.
    """
    def __init__(self, size, resample=Image.LANCZOS):
        self.size = size
        self.resample = resample

    def __call__(self, image:Image.Image) -> Image.Image:
        width, height = image.size
        if width >= height:
            new_w, new_h = self.size, max(1, round(height * self.size / width))
        else:
            new_w, new_h = max(1, round(width * self.size / height)), self.size
        return image.resize((new_w, new_h), self.resample)

    def __repr__(self):
        return f"{type(self).__name__}(size={self.size})"


class PadToSquare:
    """Pad the shorter side so the image 
    becomes centered with square canvas."""

    def __init__(self, fill:int|tuple[int,int,int]=0):
        self.fill = fill

    def __call__(self, image:Image.Image) -> Image.Image:
        width, height = image.size
        longest = max(width, height)
        pad_w = max(0, longest-width)
        pad_h = max(0, longest-height)
        border = (
            pad_w//2,
            pad_h//2,
            pad_w - pad_w//2,
            pad_h - pad_h//2)
        return ImageOps.expand(image, border=border, fill=self.fill)

    def __repr__(self):
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
        stages = geometric_augs

    stages += [
        transforms.ToTensor(),
        transforms.Normalize(list(mean), list(std))]
    
    return transforms.Compose(stages)   


# 4. GET DATASET PATHS AND LABELS
def get_img_paths_and_labels(
    base_dir:str|Path,
    extensions=(".jpg",".jpeg",".png")) -> tuple[list[Path], list[str]]:
    """Collects (paths, labels) from a 
    `base_dir/class_name/*.jpg` tree structure.

    Results are sorted. Path.glob returns entries in order similar 
    to what is in the filesystem, which can vary between machines 
    and filesystems, so an unsorted scan silently breaks reproducibility
    of any split derived from it.
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {base}")

    suffixes = {ext for ext in extensions}

    # Sorting yourself is the only way to ensure sorted order
    paths = sorted(p for p in base.glob("*/*") 
                   if p.is_file() and p.suffix.lower() in suffixes)
    
    if not paths:
        raise FileNotFoundError(f"No images with {sorted(suffixes)} found under {base}")

    labels = [p.parent.name for p in paths]
    return paths, labels


def _sanitize(name):
    """
    Match a metadata class name (from cars_meta.mat) to its folder name on disk.  
    We cannot put a slash in a directory name (the dataset's folders use a
    hyphen instead). This is shown when "Ram C/V Cargo Van Minivan 2012" become
    "Ram C-V Cargo Van Minivan 2012" in the notebook. That's the only class it 
    affects right now. Running the replacement across every name is going to be     
    a no-op for the other 195 classes, so the general form is safe to apply blindly   
    but it will be useful if in the dataset, we add another class with a slash in it.  
    """
    return name.replace("/", "-").strip()


def load_class_names(cars_meta_path, fallback_dir):
    """Return class names in canonical order.

    Prefers the ordering used in the official cars_meta.mat so integer ids are arranged in a
    way that aligns with the published class ids. This can fall back to sorted folder names, 
    which is a different ordering but self-consistent and dependency-free.
    """
    if cars_meta_path is not None:
        from scipy.io import loadmat  # import only needed here

        meta = loadmat(str(cars_meta_path))
        # Run the slash to hyphen replacement. 195 classes are no-op except one.
        return [_sanitize(cls[0]) for cls in meta["class_names"][0]]

    if fallback_dir is None:
        raise ValueError("Provide either cars_meta_path or fallback_dir")

    base = Path(fallback_dir)
    names = sorted(p.name for p in base.iterdir() if p.is_dir())
    if not names:
        raise FileNotFoundError(f"No class subdirectories found under {base}")
    return names


def build_label_map(class_names:Sequence[str]) -> dict:
    """Map class name to a 0-indexed id (PyTorch expects 0-indexed targets)."""
    mapping = {name: idx for idx, name in enumerate(class_names)}
    return mapping


# 4. DATASET SAMPLE GETTER
class CarImageDataset(Dataset):
    """
    Map-style dataset over image paths with string labels.
    `__getitem__` returns (image, label_id, label_str). The string label
    is included for error analysis and plotting.
    """

    def __init__(self, paths, str_labels, lbl_id_map, transform):
        if len(paths) != len(str_labels):
            raise ValueError(
                f"paths and str_labels have different lengths: "
                f"{len(paths)} vs. {len(str_labels)}")

        unknown = sorted(set(str_labels) - set(lbl_id_map))
        if unknown:
            preview = ", ".join(unknown[:5])
            raise KeyError(
                f"{len(unknown)} label(s) missing from the label map: {preview}")

        self.paths = paths
        self.str_labels = str_labels
        self.transform = transform

        self.cls_to_id = lbl_id_map
        self.id_to_cls = {i: c for c, i in self.cls_to_id.items()}
        self.cls_ids = [self.cls_to_id[label] for label in self.str_labels]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        label_id = self.cls_ids[idx]
        label_str = self.str_labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label_id, label_str


# 5. FACTORIES
def build_datasets(config):
    """Return (train_ds, val_ds, test_ds, label_map).

    `val_ds` is `None` when `config.val_split` is 0. This reproduces the
    notebook's setup of training on all available data (0% validation split)
    and evaluating against the test set each epoch.
    """
    train_paths, train_labels = get_img_paths_and_labels(config.train_dir)
    test_paths, test_labels = get_img_paths_and_labels(config.test_dir)

    class_names = load_class_names(config.cars_meta_path, fallback_dir=config.train_dir)
    label_map = build_label_map(class_names)

    val_paths = []
    val_labels = []
    if config.val_split > 0:
        from sklearn.model_selection import train_test_split

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths,
            train_labels,
            test_size=config.val_split,
            shuffle=True,
            stratify=train_labels,
            random_state=config.seed)

    train_tf = get_transforms(True, config.image_size, config.mean, config.std)
    eval_tf = get_transforms(False, config.image_size, config.mean, config.std)

    train_ds = CarImageDataset(train_paths, train_labels, label_map, train_tf)
    test_ds = CarImageDataset(test_paths, test_labels, label_map, eval_tf)
    val_ds = (CarImageDataset(val_paths, val_labels, label_map, eval_tf) if val_paths else None)

    return train_ds, val_ds, test_ds, label_map