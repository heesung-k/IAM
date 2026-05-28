"""DataLoader builders for CIFAR datasets."""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ..config import resolve_data_root
from .datasets import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
    default_cifar10_transforms,
    default_cifar100_transforms,
)


def _dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = False,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "persistent_workers": num_workers > 0,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **kwargs)


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    *,
    autoaugment: bool = False,
    model_type: str | None = None,
    data_root: str | None = None,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Return train/test loaders for CIFAR-10."""

    model_flag = (model_type or "WRN").lower()
    input_size = 224 if model_flag == "vit" else 32

    transform_train, transform_test = default_cifar10_transforms(
        use_autoaugment=autoaugment,
        use_cutout=autoaugment,
        input_size=input_size,
    )

    root = resolve_data_root(data_root)
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    train_loader = _dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = _dataloader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, test_loader


def get_cifar100_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    *,
    autoaugment: bool = False,
    data_root: str | None = None,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Return train/test loaders for CIFAR-100."""

    transform_train, transform_test = default_cifar100_transforms(
        use_autoaugment=autoaugment,
        use_cutout=autoaugment,
    )

    root = resolve_data_root(data_root)
    train_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    train_loader = _dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = _dataloader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, test_loader


def get_cifar10_loaders_val(
    batch_size: int = 128,
    num_workers: int = 4,
    *,
    val_split: float = 0.1,
    seed: int = 42,
    model_type: str = "WRN",
    data_root: str | None = None,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    input_size = 224 if model_type.lower() == "vit" else 32

    transform_train = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size, padding=input_size // 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    transform_val_test = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    root = resolve_data_root(data_root)
    train_full = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    val_full = datasets.CIFAR10(root=root, train=True, download=False, transform=transform_val_test)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_val_test)

    num_train = len(train_full)
    num_val = int(num_train * val_split)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_train, generator=generator)
    val_indices = indices[:num_val].tolist()
    train_indices = indices[num_val:].tolist()

    train_subset = Subset(train_full, train_indices)
    val_subset = Subset(val_full, val_indices)

    train_loader = _dataloader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = _dataloader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = _dataloader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader, test_loader


def get_cifar100_loaders_val(
    batch_size: int = 128,
    num_workers: int = 4,
    *,
    val_split: float = 0.1,
    seed: int = 42,
    data_root: str | None = None,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    transform_val_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    root = resolve_data_root(data_root)
    train_full = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    val_full = datasets.CIFAR100(root=root, train=True, download=False, transform=transform_val_test)
    test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform_val_test)

    num_train = len(train_full)
    num_val = int(num_train * val_split)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_train, generator=generator)
    train_idx = indices[num_val:].tolist()
    val_idx = indices[:num_val].tolist()

    train_subset = Subset(train_full, train_idx)
    val_subset = Subset(val_full, val_idx)

    train_loader = _dataloader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = _dataloader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = _dataloader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader, test_loader


__all__ = [
    "get_cifar10_loaders",
    "get_cifar100_loaders",
    "get_cifar10_loaders_val",
    "get_cifar100_loaders_val",
]
