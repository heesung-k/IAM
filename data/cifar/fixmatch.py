"""FixMatch data utilities built on CIFAR datasets."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ..config import resolve_data_root
from ..transforms.ssl import CIFAR10SSL, TransformFixMatch
from ..utils import set_seed, split_labeled_unlabeled
from .datasets import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD


def _loader(dataset, *, batch_size, shuffle, num_workers, drop_last=False, prefetch_factor=2):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": drop_last,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def get_fixmatch_loaders(
    batch_size: int = 64,
    num_workers: int = 8,
    *,
    num_labeled: int = 250,
    seed: int = 5,
    data_root: str | None = None,
    unlabeled_ratio: int = 7,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return FixMatch loaders for CIFAR-10."""

    set_seed(seed)
    root = resolve_data_root(data_root)

    cifar10_mean = list(CIFAR10_MEAN)
    cifar10_std = list(CIFAR10_STD)

    transform_labeled = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    transform_unlabeled = TransformFixMatch(mean=cifar10_mean, std=cifar10_std)
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    base_dataset = datasets.CIFAR10(root=root, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    labels = np.array(base_dataset.targets)
    labeled_idx, unlabeled_idx = split_labeled_unlabeled(labels, num_labeled, num_classes=10, seed=seed)

    labeled_dataset = CIFAR10SSL(
        base_dataset.data,
        labels,
        labeled_idx,
        transform=transform_labeled,
    )
    unlabeled_dataset = CIFAR10SSL(
        base_dataset.data,
        labels,
        unlabeled_idx,
        transform=transform_unlabeled,
        unlabeled_target=-1,
    )

    labeled_loader = _loader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=prefetch_factor,
    )
    unlabeled_loader = _loader(
        unlabeled_dataset,
        batch_size=batch_size * unlabeled_ratio,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=prefetch_factor,
    )
    test_loader = _loader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return labeled_loader, unlabeled_loader, test_loader


def get_fixmatch_loaders_cifar100(
    batch_size: int = 64,
    num_workers: int = 8,
    *,
    num_labeled: int = 2500,
    seed: int = 5,
    data_root: str | None = None,
    unlabeled_ratio: int = 7,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return FixMatch loaders for CIFAR-100."""

    set_seed(seed)
    root = resolve_data_root(data_root)

    cifar100_mean = list(CIFAR100_MEAN)
    cifar100_std = list(CIFAR100_STD)

    transform_labeled = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ]
    )
    transform_unlabeled = TransformFixMatch(mean=cifar100_mean, std=cifar100_std)
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ]
    )

    base_dataset = datasets.CIFAR100(root=root, train=True, download=True)
    test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    labels = np.array(base_dataset.targets)
    labeled_idx, unlabeled_idx = split_labeled_unlabeled(labels, num_labeled, num_classes=100, seed=seed)

    labeled_dataset = CIFAR10SSL(
        base_dataset.data,
        labels,
        labeled_idx,
        transform=transform_labeled,
    )
    unlabeled_dataset = CIFAR10SSL(
        base_dataset.data,
        labels,
        unlabeled_idx,
        transform=transform_unlabeled,
        unlabeled_target=-1,
    )

    labeled_loader = _loader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=prefetch_factor,
    )
    unlabeled_loader = _loader(
        unlabeled_dataset,
        batch_size=batch_size * unlabeled_ratio,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=prefetch_factor,
    )
    test_loader = _loader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return labeled_loader, unlabeled_loader, test_loader


def get_fixmatch_loaders_val(
    batch_size: int = 64,
    num_workers: int = 10,
    *,
    num_labeled: int = 250,
    val_ratio: float = 0.1,
    seed: int = 5,
    data_root: str | None = None,
    unlabeled_ratio: int = 7,
    prefetch_factor: int = 2,
):
    """Return FixMatch loaders with extra validation split for CIFAR-10."""

    set_seed(seed)
    root = resolve_data_root(data_root)

    cifar10_mean = list(CIFAR10_MEAN)
    cifar10_std = list(CIFAR10_STD)

    transform_labeled = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    transform_unlabeled = TransformFixMatch(mean=cifar10_mean, std=cifar10_std)
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    base_dataset = datasets.CIFAR10(root=root, train=True, download=True)
    labels = np.array(base_dataset.targets)
    num_samples = len(labels)

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    val_size = int(num_samples * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_labels = labels[train_indices]
    labeled_idx, unlabeled_idx = split_labeled_unlabeled(train_labels, num_labeled, num_classes=10, seed=seed)

    labeled_idx = train_indices[labeled_idx]
    unlabeled_idx = train_indices[unlabeled_idx]

    labeled_dataset = CIFAR10SSL(
        base_dataset.data,
        labels,
        labeled_idx,
        transform=transform_labeled,
    )
    unlabeled_dataset = CIFAR10SSL(
        base_dataset.data,
        labels,
        unlabeled_idx,
        transform=transform_unlabeled,
        unlabeled_target=-1,
    )

    val_dataset = Subset(
        datasets.CIFAR10(root=root, train=True, download=True, transform=transform_test),
        val_indices.tolist(),
    )
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    labeled_loader = _loader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=prefetch_factor,
    )
    unlabeled_loader = _loader(
        unlabeled_dataset,
        batch_size=batch_size * unlabeled_ratio,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=prefetch_factor,
    )
    val_loader = _loader(
        val_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = _loader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return labeled_loader, unlabeled_loader, val_loader, test_loader


__all__ = [
    "get_fixmatch_loaders",
    "get_fixmatch_loaders_cifar100",
    "get_fixmatch_loaders_val",
]
