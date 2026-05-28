"""Helpers that construct per-rank data loaders for distributed training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from data import get_datasets, get_imagenet_datasets

from .ddp_utils import DDPState


@dataclass
class DataLoaders:
    train: DataLoader
    eval: DataLoader
    per_device_batch_size: int
    num_classes: int
    input_channels: int
    weight_decay: float


def _resolve_dataset(dataset_name: str, data_dir: str) -> Tuple[Dataset, Dataset, int, int, float]:
    if dataset_name == "imagenet":
        train_dataset, eval_dataset = get_imagenet_datasets(data_dir=data_dir)
        return train_dataset, eval_dataset, 1000, 3, 1e-4

    if dataset_name in {"cifar10", "cifar100", "mnist", "fashion_mnist"}:
        train_dataset, eval_dataset = get_datasets(dataset_name)
        num_classes = 10 if dataset_name in {"cifar10", "mnist", "fashion_mnist"} else 100
        input_channels = 1 if dataset_name in {"mnist", "fashion_mnist"} else 3
        if dataset_name in {"cifar10", "cifar100"}:
            weight_decay = 5e-4
        elif dataset_name in {"mnist", "fashion_mnist"}:
            weight_decay = 1e-3
        else:
            weight_decay = 1e-4
        return train_dataset, eval_dataset, num_classes, input_channels, weight_decay

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    sampler: Optional[Iterable[int]],
    shuffle: bool,
    args,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle if sampler is None else False),  # sampler 있으면 shuffle 무시
        num_workers=num_workers,
        prefetch_factor=(args.prefetch_factor if num_workers > 0 else None),
        pin_memory=args.pin_memory,
        pin_memory_device=args.pin_memory_device,  # PyTorch>=2.x에서만
        persistent_workers=(args.persistent_workers and num_workers > 0),
        drop_last = drop_last
    )


def build_distributed_dataloaders(args, ddp_state: DDPState) -> DataLoaders:
    """Create per-rank loaders for training and evaluation."""

    train_dataset, eval_dataset, num_classes, channels, weight_decay = _resolve_dataset(
        args.dataset, getattr(args, "data_dir", "")
    )

    per_device_batch = max(1, args.batch_size // max(ddp_state.world_size, 1))
    if args.batch_size % max(ddp_state.world_size, 1) != 0 and ddp_state.is_primary:
        print(
            f"Warning: global batch size {args.batch_size} not divisible by world size {ddp_state.world_size}."
            f" Using per-device batch size {per_device_batch}."
        )

    if ddp_state.is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=ddp_state.world_size, rank=ddp_state.rank, shuffle=True
        )
        eval_sampler = DistributedSampler(
            eval_dataset, num_replicas=ddp_state.world_size, rank=ddp_state.rank, shuffle=False
        )
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = _build_loader(
        train_dataset,
        batch_size=per_device_batch,
        num_workers=args.num_workers,
        sampler=train_sampler,
        shuffle=True,
        args=args,
        drop_last=True,
    )

    eval_loader = _build_loader(
        eval_dataset,
        batch_size=per_device_batch,
        num_workers=args.num_workers,
        sampler=eval_sampler,
        shuffle=False,
        args=args,
        drop_last=False,
    )

    return DataLoaders(
        train=train_loader,
        eval=eval_loader,
        per_device_batch_size=per_device_batch,
        num_classes=num_classes,
        input_channels=channels,
        weight_decay=weight_decay,
    )
