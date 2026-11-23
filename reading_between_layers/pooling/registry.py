"""Mapping from pooling-strategy name to its class."""

from __future__ import annotations

from .strategies import (
    AttentionPooling,
    CLSPooling,
    LastTokenPooling,
    MaxPooling,
    MeanPooling,
    SBERTPairPooling,
)

POOLING_REGISTRY: dict[str, type] = {
    "cls": CLSPooling,
    "mean": MeanPooling,
    "last_token": LastTokenPooling,
    "max": MaxPooling,
    "attention": AttentionPooling,
    "sbert_pair": SBERTPairPooling,
}
