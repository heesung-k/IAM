"""Utility helpers shared across data pipelines."""
from __future__ import annotations

import random
from typing import Iterable, Tuple

import numpy as np
import torch


def set_seed(seed: int | None) -> None:
    """Seed ``random``, ``numpy`` and torch (CPU/GPU) RNGs when provided."""

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_labeled_unlabeled(
    labels: Iterable[int] | np.ndarray,
    num_labeled: int,
    *,
    num_classes: int = 10,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create class-balanced labeled/unlabeled splits for semi-supervision."""

    labels_arr = np.asarray(labels)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    labeled_idx: list[int] = []
    unlabeled_idx: list[int] = []

    per_class = num_labeled // num_classes
    for class_id in range(num_classes):
        class_indices = np.where(labels_arr == class_id)[0]
        np.random.shuffle(class_indices)
        labeled_idx.extend(class_indices[:per_class])
        unlabeled_idx.extend(class_indices[per_class:])

    return np.array(labeled_idx), np.array(unlabeled_idx)


__all__ = [
    "set_seed",
    "split_labeled_unlabeled",
]
