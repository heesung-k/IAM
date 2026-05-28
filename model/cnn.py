"""Utility to build the lightweight 6-layer CNN baseline."""
from __future__ import annotations

import torch.nn as nn


def CNN(channels: int = 3, dence: int = 8 * 8 * 64, dropout: float = 0.0, labels: int = 10) -> nn.Sequential:
    if dropout > 0.0:
        raise Warning("there is no dropout layer for 6CNN")
    return nn.Sequential(
        nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(dence, 100, bias=True),
        nn.ReLU(),
        nn.Linear(100, labels, bias=True),
    )


__all__ = ["CNN"]
