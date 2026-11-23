"""Learnable scalar weighting over cached layer embeddings."""

from __future__ import annotations

import pandas as pd
import torch
from torch import Tensor, nn

from ..tasks.compatibility import TaskType
from .base import ProbeBase

SINGLE_LAYER_AGGREGATION = "single_layer"
LEARNABLE_WEIGHTING_AGGREGATION = "learnable_weighting"
LAYER_AGGREGATION_COLUMN = "layer_aggregation"

# Keep this far away from real model layers and the sentence-transformer pooled
# pseudo-layer (-1). It is a result sentinel, not an embedding layer.
LEARNABLE_WEIGHTING_LAYER_INDEX = -10_000

TRAINABLE_LAYER_WEIGHTING_TASKS = frozenset(
    {
        TaskType.CLASSIFICATION,
        TaskType.PAIR_CLASSIFICATION,
        TaskType.MULTIPLE_CHOICE_QA,
        TaskType.STS,
    }
)

TRAINABLE_LAYER_WEIGHTING_PROBES = frozenset({"linear", "nonlinear"})


class LayerWeightedProbe(ProbeBase):
    """Softmax scalar mix over layers followed by an existing probe head.

    The layer dimension is always the second-to-last feature dimension:
    ``(n, layers, hidden)`` for normal pooled features and
    ``(n, options, layers, hidden)`` for per-option multiple-choice features.
    """

    def __init__(self, base_probe: ProbeBase, num_layers: int) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        super().__init__(
            input_dim=base_probe.input_dim,
            output_dim=base_probe.output_dim,
            task_type=base_probe.task_type,
            config=base_probe.config,
        )
        self.base_probe = base_probe
        self.num_layers = int(num_layers)
        self.layer_logits = nn.Parameter(torch.zeros(self.num_layers))

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim < 3:
            raise ValueError(
                "LayerWeightedProbe expects a layer dimension before hidden; "
                f"got shape {tuple(embeddings.shape)}"
            )
        if embeddings.size(-2) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} layers on dim -2, got {embeddings.size(-2)}"
            )
        weights = torch.softmax(self.layer_logits, dim=0).to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        view_shape = [1] * embeddings.ndim
        view_shape[-2] = self.num_layers
        mixed = (embeddings * weights.view(*view_shape)).sum(dim=-2)
        return self.base_probe(mixed)

    def layer_weights(self) -> dict[int, float]:
        """Return zero-based temporary indices; caller remaps to real layer ids."""
        with torch.no_grad():
            weights = torch.softmax(self.layer_logits, dim=0).detach().cpu().numpy()
        return {int(i): float(value) for i, value in enumerate(weights)}


def layer_aggregation_series(df: pd.DataFrame) -> pd.Series:
    """Return aggregation labels, treating old rows as single-layer results."""
    if LAYER_AGGREGATION_COLUMN not in df.columns:
        return pd.Series(SINGLE_LAYER_AGGREGATION, index=df.index, dtype="object")
    return df[LAYER_AGGREGATION_COLUMN].fillna(SINGLE_LAYER_AGGREGATION).astype(str)


def single_layer_probe_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that represent ordinary one-layer probe results."""
    if df.empty:
        return df.copy()
    mask = layer_aggregation_series(df) == SINGLE_LAYER_AGGREGATION
    return df.loc[mask].copy()


def learnable_weighting_probe_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that represent learned all-layer weighting results."""
    if df.empty:
        return df.copy()
    mask = layer_aggregation_series(df) == LEARNABLE_WEIGHTING_AGGREGATION
    return df.loc[mask].copy()
