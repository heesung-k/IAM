"""ImageNet dataset helpers."""
from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
from ..config import resolve_data_root
from torchvision.io import read_image, ImageReadMode
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def read_image_loader(path: str):
    # ImageNet은 RGB 이미지를 기대하므로 mode를 명시
    return read_image(path, mode=ImageReadMode.RGB)

def _get_transforms(image_size: int):
    train_tf = v2.Compose([
        v2.RandomResizedCrop(size=(image_size, image_size), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
    ])
    val_tf = v2.Compose([
        v2.Resize(size=256, antialias=True),
        v2.CenterCrop(size=image_size),
        v2.ToImage(),
    ])
    return train_tf, val_tf

def get_imagenet_datasets(data_dir: str | None = None, image_size: int = 224):
    root = resolve_data_root(data_dir)
    train_tf, val_tf = _get_transforms(image_size)
    
    train_ds = datasets.ImageNet(root=root, split="train", transform=train_tf, loader=read_image_loader)
    val_ds = datasets.ImageNet(root=root, split="val", transform=val_tf, loader=read_image_loader)
    return train_ds, val_ds

def get_imagenet_loaders(
    *,
    data_dir: str | None = None,
    batch_size: int = 256,
    num_workers: int = 8,
    image_size: int = 224,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader]:

    train_ds, val_ds = get_imagenet_datasets(data_dir=data_dir, image_size=image_size)

    def _loader(dataset, *, shuffle: bool) -> DataLoader:
        kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(dataset, **kwargs)

    return _loader(train_ds, shuffle=True), _loader(val_ds, shuffle=False)


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "get_imagenet_loaders",
    "get_imagenet_datasets",
]
