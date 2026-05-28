"""Central registry for dataset factories."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

from torch.utils.data import DataLoader, Dataset

from .cifar.datasets import get_cifar10_dataset, get_cifar100_dataset
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
from .imagenet.loaders import get_imagenet_datasets, get_imagenet_loaders
from .svhn import get_svhn_loaders

DatasetFactory = Callable[..., Tuple[Dataset, Dataset]]
LoaderFactory = Callable[..., Tuple[DataLoader, ...]]

_DATASET_FACTORIES: Dict[str, DatasetFactory] = {
    "cifar10": get_cifar10_dataset,
    "cifar100": get_cifar100_dataset,
    "imagenet": get_imagenet_datasets,
}

_LOADER_FACTORIES: Dict[str, LoaderFactory] = {
    "cifar10": get_cifar10_loaders,
    "cifar100": get_cifar100_loaders,
    "cifar10_val": get_cifar10_loaders_val,
    "cifar100_val": get_cifar100_loaders_val,
    "cifar10_fixmatch": get_fixmatch_loaders,
    "cifar100_fixmatch": get_fixmatch_loaders_cifar100,
    "cifar10_fixmatch_val": get_fixmatch_loaders_val,
    "fashion_mnist": get_fashion_mnist_loaders,
    "svhn": get_svhn_loaders,
    "imagenet": get_imagenet_loaders,
}


def get_dataset(name: str, /, **kwargs):
    key = name.lower()
    if key not in _DATASET_FACTORIES:
        raise ValueError(f"Unknown dataset: {name}")
    return _DATASET_FACTORIES[key](**kwargs)


def get_loader(name: str, /, **kwargs):
    key = name.lower()
    if key not in _LOADER_FACTORIES:
        raise ValueError(f"Unknown loader: {name}")
    return _LOADER_FACTORIES[key](**kwargs)


__all__ = [
    "get_dataset",
    "get_loader",
]
