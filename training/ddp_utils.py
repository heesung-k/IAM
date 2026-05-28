"""Distributed training helpers for torch.distributed."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


def _infer_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


@dataclass
class DDPState:
    """Small container that stores the distributed runtime information."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


class DDPEnvironment(AbstractContextManager):
    """Context manager that initialises and tears down torch.distributed."""

    def __init__(self, backend: str = "nccl") -> None:
        self.backend = backend
        self._state: Optional[DDPState] = None
        self._initialised_here = False

    def __enter__(self) -> DDPState:
        if not dist.is_available():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._state = DDPState(rank=0, world_size=1, local_rank=0, device=device)
            return self._state

        if not dist.is_initialized():
            dist.init_process_group(backend=self.backend)
            self._initialised_here = True

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = _infer_device(local_rank)

        self._state = DDPState(rank=rank, world_size=world_size, local_rank=local_rank, device=device)
        return self._state

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._initialised_here and dist.is_initialized():
            dist.destroy_process_group()

    @property
    def state(self) -> DDPState:
        if self._state is None:
            raise RuntimeError("DDPEnvironment is not initialised. Use it as a context manager or call __enter__().")
        return self._state


def barrier_if_distributed() -> None:
    """Synchronise all processes if DDP is active."""

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce *tensor* across workers by summing when DDP is active."""

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
