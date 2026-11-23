"""Pooling strategies that reduce per-layer token states to a single vector."""

from __future__ import annotations

from typing import Any


def get_pooling(name: str, **kwargs: Any):
    """Instantiate the pooling strategy registered under ``name``.

    Args:
        name: registry key. ``AttentionPooling`` requires ``hidden_size``.
        **kwargs: forwarded to the strategy constructor.

    Returns:
        A new strategy instance.

    Raises:
        KeyError: if ``name`` is not registered.
    """
    from .registry import POOLING_REGISTRY

    cls = POOLING_REGISTRY[name]
    return cls(**kwargs)


def __getattr__(name: str):
    if name == "POOLING_REGISTRY":
        from .registry import POOLING_REGISTRY

        return POOLING_REGISTRY
    raise AttributeError(name)
