"""Semantic textual similarity task with Spearman and Pearson evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .base import Task, TaskSpec
from .compatibility import TaskType
from .metrics import mean_squared_error, pearson, spearman

if TYPE_CHECKING:
    from ..datasets.base import DatasetSample


class STSTask(Task):
    """STS regression task; primary metric is Spearman correlation."""

    def __init__(self) -> None:
        self.spec = TaskSpec(
            task_type=TaskType.STS,
            pooling_default="sbert_pair",
            pooling_options=["mean", "sbert_pair"],
            metric_primary="spearman",
            metrics_secondary=["pearson", "mse"],
            allowed_families=["encoder", "decoder", "encoder_decoder", "ssm"],
            probe_default="linear",
            probe_types_allowed=["linear", "nonlinear", "lora", "full_ft"],
            needs_labels_for_geometry=False,
        )

    def build_features(
        self,
        samples: list[DatasetSample],
        pooled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        labels = torch.tensor([float(s.label) for s in samples], dtype=torch.float32)
        return pooled, labels

    def evaluate(self, predictions: Tensor, labels: Tensor) -> dict[str, float]:
        return {
            "spearman": spearman(predictions, labels),
            "pearson": pearson(predictions, labels),
            "mse": mean_squared_error(predictions, labels),
        }
