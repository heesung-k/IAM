"""Model architectures used throughout the IAM project."""
from __future__ import annotations

from .cnn import CNN
from .wideresnet import WideResNet

__all__ = ["WideResNet", "CNN"]
