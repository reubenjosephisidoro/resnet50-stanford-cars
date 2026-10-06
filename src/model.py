"""
ResNet50 architecture for Stanford Cars classification.
Loads a pretrained model (primarily ResNet50) from torchvision then replaces 
the final fully-connected layer so it matches `num_classes` logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

__all__ = ["build_model"]


def build_model(num_classes, device: torch.device|str="cpu") -> nn.Module:
    """Load a pretrained model (ResNet50) and replace the final layer.

    Args:
        num_classes: Number of classes to output (196 for Stanford Cars).
        device: The device on which to place the model (cpu, cuda, or a torch.device).

    Returns:
        A ResNet50 model with the fc layer replaced. The caller is responsible
        for DataParallel wrapping and any layer freezing.
    """
    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Replace the final classification layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move to device
    model = model.to(device)

    return model


if __name__ == "__main__":
    # Smoke test: build the model and check output shape
    model = build_model(num_classes=196, device="cpu")
    print(f"Model: {type(model).__name__}")
    print(f"Final layer: {model.fc}")

    # Dummy forward pass
    dummy_input = torch.randn(2, 3, 448, 448)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Expected: torch.Size([2, 196])")