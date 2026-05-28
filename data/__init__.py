"""Public data API for IAM experiments."""
from __future__ import annotations

from .config import DATA_ROOT_ENV, DEFAULT_DATA_ROOT, resolve_data_root
from .cifar.datasets import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
    get_cifar10_dataset,
    get_cifar100_dataset,
)
from .cifar.fixmatch import (
    get_fixmatch_loaders,
    get_fixmatch_loaders_cifar100,
    get_fixmatch_loaders_val,
)
from .cifar.loaders import (
    get_cifar10_loaders,
    get_cifar10_loaders_val,
    get_cifar100_loaders,
    get_cifar100_loaders_val,
)
from .fashion_mnist import get_fashion_mnist_loaders
from .imagenet.loaders import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_imagenet_datasets,
    get_imagenet_loaders,
)
from .registry import get_dataset, get_loader
from .svhn import get_svhn_loaders
from .transforms.cutout import Cutout
from .transforms.ssl import CIFAR10SSL, TransformFixMatch
from .utils import set_seed, split_labeled_unlabeled


def get_datasets(dataset_name: str, /, **kwargs):
    """Backward compatible dataset accessor for common benchmarks."""

    key = dataset_name.lower()
    if key == "cifar10":
        return get_cifar10_dataset(**kwargs)
    if key == "cifar100":
        return get_cifar100_dataset(**kwargs)
    if key == "imagenet":
        return get_imagenet_datasets(**kwargs)

    raise ValueError(f"Unsupported dataset for get_datasets: {dataset_name}")

__all__ = [
    "DATA_ROOT_ENV",
    "DEFAULT_DATA_ROOT",
    "resolve_data_root",
    "get_datasets",
    "get_dataset",
    "get_loader",
    "get_cifar10_dataset",
    "get_cifar100_dataset",
    "get_cifar10_loaders",
    "get_cifar100_loaders",
    "get_cifar10_loaders_val",
    "get_cifar100_loaders_val",
    "get_fixmatch_loaders",
    "get_fixmatch_loaders_cifar100",
    "get_fixmatch_loaders_val",
    "get_fashion_mnist_loaders",
    "get_svhn_loaders",
    "get_imagenet_loaders",
    "get_imagenet_datasets",
    "set_seed",
    "split_labeled_unlabeled",
    "Cutout",
    "TransformFixMatch",
    "CIFAR10SSL",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
