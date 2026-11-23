"""Shared tensor utilities for masked pooling and dtype conversion."""

from __future__ import annotations

import torch
from torch import Tensor

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def masked_mean(x: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    """Mean of ``x`` along ``dim`` ignoring positions where ``mask`` is zero.

    Args:
        x: tensor of shape (batch, seq, hidden).
        mask: tensor of shape (batch, seq) with 1 for valid positions, 0 for pads.
        dim: sequence dimension; defaults to 1.

    Returns:
        Tensor of shape (batch, hidden).
    """
    mask = mask.unsqueeze(-1).to(x.dtype)
    summed = (x * mask).sum(dim=dim)
    counts = mask.sum(dim=dim).clamp(min=1)
    return summed / counts


def masked_max(x: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    """Max of ``x`` along ``dim`` with pad positions filled to negative infinity."""
    bool_mask = mask.unsqueeze(-1).to(torch.bool)
    filled = x.masked_fill(~bool_mask, float("-inf"))
    return filled.max(dim=dim).values


def gather_last_token(x: Tensor, attention_mask: Tensor) -> Tensor:
    """Return the hidden state at the last non-pad position for each row."""
    lengths = attention_mask.sum(dim=1) - 1
    lengths = lengths.clamp(min=0).to(torch.long)
    idx = lengths.view(-1, 1, 1).expand(-1, 1, x.size(-1))
    return x.gather(1, idx).squeeze(1)


def to_dtype(x: Tensor, dtype: str) -> Tensor:
    """Cast ``x`` to a torch dtype identified by string."""
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"unsupported dtype: {dtype}")
    return x.to(_DTYPE_MAP[dtype])
