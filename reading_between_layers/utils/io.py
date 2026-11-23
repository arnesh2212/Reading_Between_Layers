"""Atomic file writes and tensor save/load helpers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor


@contextmanager
def atomic_write_path(path: Path) -> Iterator[Path]:
    """Yield a temporary path; rename it to ``path`` on successful exit.

    The temporary file is removed if the context body raises, so no partial
    artifact remains in the target directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        yield tmp
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    else:
        os.replace(tmp, path)


def save_tensor(tensor: Tensor, path: Path) -> None:
    """Atomically save a tensor to ``path``."""
    with atomic_write_path(path) as tmp:
        torch.save(tensor, tmp)


def load_tensor(path: Path, *, mmap: bool = False) -> Tensor:
    """Load a tensor previously written by :func:`save_tensor`."""
    return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
