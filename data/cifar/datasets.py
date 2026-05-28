"""Datasets and transforms for CIFAR family."""
from __future__ import annotations
import torch
from typing import Tuple

from torchvision import datasets, transforms
from torchvision.transforms import v2

from ..config import resolve_data_root
from ..transforms.cutout import Cutout

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def default_cifar10_transforms(
    *,
    use_autoaugment: bool = False,
    use_cutout: bool = False,
    input_size: int = 32,
):
    augment = [
        v2.RandomCrop(input_size, padding=input_size // 8),
        v2.RandomHorizontalFlip(),
    ]
    if use_autoaugment:
        from torchvision.transforms import AutoAugment, AutoAugmentPolicy

        augment.append(AutoAugment(policy=AutoAugmentPolicy.CIFAR10))
    
    augment.append(v2.ToImage()) # Keep as uint8
    # augment.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)) # Removed

    if use_cutout:
        # Calculate mean fill value (uint8)
        fill = tuple(int(x * 255) for x in CIFAR10_MEAN)
        augment.append(Cutout(size=8, fill=fill))

    transform_train = v2.Compose(augment)
    transform_test = v2.Compose(
        [
            v2.Resize(input_size),
            v2.ToImage(),
            # v2.Normalize(CIFAR10_MEAN, CIFAR10_STD), # Removed
        ]
    )

    return transform_train, transform_test


def get_cifar10_dataset(
    *,
    data_root: str | None = None,
    input_size: int = 32,
    use_autoaugment: bool = False,
    use_cutout: bool = False,
) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Return train/test CIFAR-10 datasets with standard transforms."""

    root = resolve_data_root(data_root)
    transform_train, transform_test = default_cifar10_transforms(
        use_autoaugment=use_autoaugment,
        use_cutout=use_cutout,
        input_size=input_size,
    )

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset

class FastCIFAR100(datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # numpy (N,H,W,C) uint8 -> torch (N,C,H,W) uint8 (한 번만)
        self.data = torch.from_numpy(self.data).permute(0, 3, 1, 2).contiguous()

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]  # img: (C,H,W) uint8
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def default_cifar100_transforms(*, use_autoaugment: bool = False, use_cutout: bool = False):
    augment = [
        # v2.Resize(32),
        v2.RandomCrop(32, padding=32 // 8),
        v2.RandomHorizontalFlip(),
    ]
    if use_autoaugment:
        from torchvision.transforms import AutoAugmentPolicy

        augment.append(v2.AutoAugment(policy=AutoAugmentPolicy.CIFAR10))
    
    # augment.extend([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]) # Removed

    if use_cutout:
        # Calculate mean fill value (uint8)
        fill = tuple(int(x * 255) for x in CIFAR100_MEAN)
        augment.append(Cutout(size=8, fill=fill))

    transform_train = v2.Compose(augment)
    transform_test = None
    return transform_train, transform_test


def get_cifar100_dataset(
    *,
    data_root: str | None = None,
    use_autoaugment: bool = False,
    use_cutout: bool = False,
) -> Tuple[datasets.CIFAR100, datasets.CIFAR100]:
    root = resolve_data_root(data_root)
    transform_train, transform_test = default_cifar100_transforms(
        use_autoaugment=use_autoaugment,
        use_cutout=use_cutout,
    )

    train_dataset = FastCIFAR100(root=root, train=True, download=True, transform=transform_train)
    test_dataset = FastCIFAR100(root=root, train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset


__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "default_cifar10_transforms",
    "default_cifar100_transforms",
    "get_cifar10_dataset",
    "get_cifar100_dataset",
]
