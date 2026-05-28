"""Fashion-MNIST loader helpers."""
from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import resolve_data_root

FASHION_MEAN = (0.2860,)
FASHION_STD = (0.3530,)


def get_fashion_mnist_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    *,
    data_root: str | None = None,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Return train/test loaders for Fashion-MNIST."""

    root = resolve_data_root(data_root)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MEAN, FASHION_STD),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MEAN, FASHION_STD),
        ]
    )

    train_dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform_test)

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, **kwargs)

    return train_loader, test_loader


__all__ = [
    "get_fashion_mnist_loaders",
]
