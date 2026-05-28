"""Backward-compatible shim for IAM optimizers."""

from __future__ import annotations

from .iam_d import (
    IAM_D,
    IAM_DE,
    SimclrLoss_IAM,
    inconsistencyLoss,
    inconsistency_FixMatch,
    inconsistency_semi,
)
from .iam_s import IAM_S, IAM_S_AMP, IAM_S_ssl

__all__ = [
    "IAM_D",
    "IAM_DE",
    "IAM_S",
    "IAM_S_AMP",
    "IAM_S_ssl",
    "SimclrLoss_IAM",
    "inconsistencyLoss",
    "inconsistency_FixMatch",
    "inconsistency_semi",
]
