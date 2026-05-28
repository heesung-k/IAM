"""Custom optimizers and losses used across IAM experiments."""
from __future__ import annotations

from .sam import ASAMLoss, FlatMatchLoss, SAMLoss, SAM
from .iam import (
    IAM_D,
    IAM_DE,
    IAM_S,
    IAM_S_AMP,
    IAM_S_ssl,
    SimclrLoss_IAM,
    inconsistencyLoss,
    inconsistency_FixMatch,
    inconsistency_semi,
)

__all__ = [
    "SAMLoss",
    "ASAMLoss",
    "FlatMatchLoss",
    "inconsistencyLoss",
    "inconsistency_FixMatch",
    "inconsistency_semi",
    "IAM_D",
    "IAM_DE",
    "IAM_S",
    "IAM_S_AMP",
    "IAM_S_ssl",
    "SimclrLoss_IAM",
    "SAM",
]
