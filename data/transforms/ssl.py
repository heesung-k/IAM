"""Semi-supervised augmentation utilities (FixMatch style)."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:  # torchvision >= 0.9
    from torchvision.transforms import RandAugment
except ImportError:  # pragma: no cover - fallback when RandAugment is unavailable
    RandAugment = None  # type: ignore

try:  # torchvision >= 0.10
    from torchvision.transforms import AutoAugment, AutoAugmentPolicy
except ImportError:  # pragma: no cover
    AutoAugment = AutoAugmentPolicy = None  # type: ignore


class TransformFixMatch:
    """Return weak/strong augmented pairs for FixMatch-style training."""

    def __init__(
        self,
        *,
        mean: Sequence[float],
        std: Sequence[float],
        crop_size: int = 32,
        padding: int = 4,
        n_ops: int = 2,
        magnitude: int = 10,
    ) -> None:
        normalize = transforms.Normalize(mean, std)
        common = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size, padding=padding, padding_mode="reflect")]

        self.weak = transforms.Compose(common + [transforms.ToTensor(), normalize])

        strong_ops = []
        if RandAugment is not None:
            strong_ops.append(RandAugment(num_ops=n_ops, magnitude=magnitude))
        elif AutoAugment is not None and AutoAugmentPolicy is not None:
            strong_ops.append(AutoAugment(policy=AutoAugmentPolicy.CIFAR10))
        else:  # last resort colour jitter to ensure "strong" difference
            strong_ops.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))

        self.strong = transforms.Compose(common + strong_ops + [transforms.ToTensor(), normalize])

    def __call__(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.weak(image), self.strong(image)


class CIFAR10SSL(Dataset):
    """Dataset wrapper that supports labeled/unlabeled FixMatch sampling."""

    def __init__(
        self,
        data: np.ndarray,
        targets: Iterable[int] | np.ndarray,
        indices: Iterable[int],
        *,
        transform=None,
        target_transform=None,
        unlabeled_target: int | None = None,
    ) -> None:
        self.data = data[indices]
        self.targets = np.array(targets)[indices]
        self.transform = transform
        self.target_transform = target_transform
        self.unlabeled_target = unlabeled_target

        self._default_to_tensor = transforms.ToTensor()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)

    def __getitem__(self, index: int):
        img = self.data[index]
        target = int(self.targets[index])

        img = Image.fromarray(img)

        if self.transform is not None:
            result = self.transform(img)
        else:
            result = self._default_to_tensor(img)

        if isinstance(result, tuple):
            output = result
        else:
            output = result

        if self.unlabeled_target is not None and isinstance(output, tuple):
            target = self.unlabeled_target

        if self.target_transform is not None:
            target = self.target_transform(target)

        return output, target


__all__ = [
    "TransformFixMatch",
    "CIFAR10SSL",
]
