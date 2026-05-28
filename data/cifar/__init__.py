"""CIFAR helpers namespace."""
from __future__ import annotations

from .datasets import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
    default_cifar10_transforms,
    default_cifar100_transforms,
    get_cifar10_dataset,
    get_cifar100_dataset,
)
from .fixmatch import (
    get_fixmatch_loaders,
    get_fixmatch_loaders_cifar100,
    get_fixmatch_loaders_val,
)
from .loaders import (
    get_cifar10_loaders,
    get_cifar10_loaders_val,
    get_cifar100_loaders,
    get_cifar100_loaders_val,
)

__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "default_cifar10_transforms",
    "default_cifar100_transforms",
    "get_cifar10_dataset",
    "get_cifar100_dataset",
    "get_cifar10_loaders",
    "get_cifar100_loaders",
    "get_cifar10_loaders_val",
    "get_cifar100_loaders_val",
    "get_fixmatch_loaders",
    "get_fixmatch_loaders_cifar100",
    "get_fixmatch_loaders_val",
]
