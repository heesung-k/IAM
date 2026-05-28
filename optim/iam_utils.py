"""Shared helpers for IAM optimizers."""

from __future__ import annotations

import torch
from torch import nn

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _disable_running_stats(model: nn.Module) -> None:
    """Temporarily freeze BatchNorm running statistics."""

    def _disable(module: nn.Module) -> None:
        if isinstance(module, _BN_TYPES):
            if not hasattr(module, "backup_momentum"):
                module.backup_momentum = module.momentum
            module.momentum = 0.0

    model.apply(_disable)


def _enable_running_stats(model: nn.Module) -> None:
    """Restore BatchNorm running statistics after a temporary freeze."""

    def _enable(module: nn.Module) -> None:
        if isinstance(module, _BN_TYPES) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def _check_finite(tag: str, tensor: torch.Tensor, raise_err: bool = False) -> bool:
    """Log or raise when the tensor contains NaN or Inf."""
    if not torch.isfinite(tensor).all():
        message = f"[{tag}] found NaN/Inf"
        if raise_err:
            raise FloatingPointError(message)
        print(message)
        return False
    return True


__all__ = ["_disable_running_stats", "_enable_running_stats", "_check_finite"]
