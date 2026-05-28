"""Configuration helpers for dataset paths and shared constants."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_ROOT = "/home/dataset/"
DATA_ROOT_ENV = "IAM_DATA_DIR"


def resolve_data_root(override: str | None = None) -> str:
    """Return the canonical dataset root directory.

    Resolution priority:
    1. Explicit ``override`` argument if provided.
    2. ``IAM_DATA_DIR`` environment variable (expanded).
    3. The legacy default path ``/home/dataset/``.
    """

    candidate = override or os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT)
    return str(Path(candidate).expanduser())


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DATA_ROOT_ENV",
    "resolve_data_root",
]
