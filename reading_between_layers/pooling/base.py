"""Pooling strategy interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class PoolingStrategy(Protocol):
    """Reduce a per-token sequence of hidden states to a fixed-size vector.

    Implementations carry a ``name`` matching their registry key. Stateful
    strategies (e.g., attention pooling) set ``requires_training = True`` so
    that linear/nonlinear probes can refuse them.
    """

    name: str
    requires_training: bool

    def pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Return ``(batch, hidden)`` from ``(batch, seq, hidden)`` and mask ``(batch, seq)``."""
        ...
