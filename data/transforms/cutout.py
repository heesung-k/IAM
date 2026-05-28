"""Cutout augmentation utilities."""
from __future__ import annotations

import torch


class Cutout:
    """Randomly masks a square region of an image."""

    def __init__(self, size: int = 16, p: float = 0.5, fill: int | tuple = 0) -> None:
        self.size = size
        self.half_size = size // 2
        self.p = p
        self.fill = fill

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return image

        channels, height, width = image.shape
        left = torch.randint(-self.half_size, width - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, height - self.half_size, [1]).item()
        right = min(width, left + self.size)
        bottom = min(height, top + self.size)

        # fill value handling varies by implementation, here we simply assign
        # If fill is a tuple/list, we might need channel-wise assignment, but standard Cutout often uses 0 or mean.
        # For simplicity and speed in this custom implementation:
        if isinstance(self.fill, (int, float)):
             image[:, max(0, left): right, max(0, top): bottom] = self.fill
        else:
             # Assume fill is per-channel tuple/list
             for c in range(channels):
                 image[c, max(0, left): right, max(0, top): bottom] = self.fill[c]
        return image


__all__ = ["Cutout"]
