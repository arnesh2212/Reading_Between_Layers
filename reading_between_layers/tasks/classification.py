"""Single-text classification task: head shape, loss, and metric calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .base import Task, TaskSpec
from .compatibility import TaskType
from .metrics import accuracy, macro_f1

if TYPE_CHECKING:
    from ..datasets.base import DatasetSample


class ClassificationTask(Task):
    """Single-text classification with accuracy as the primary metric."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.spec = TaskSpec(
            task_type=TaskType.CLASSIFICATION,
            pooling_default="cls",
            pooling_options=["cls", "mean", "max", "attention"],
            metric_primary="accuracy",
            metrics_secondary=["macro_f1"],
            allowed_families=["encoder", "decoder", "encoder_decoder", "ssm"],
            probe_default="linear",
            probe_types_allowed=["linear", "nonlinear", "lora", "full_ft"],
            needs_labels_for_geometry=True,
        )

    def build_features(
        self,
        samples: list[DatasetSample],
        pooled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(pooled, labels)``; labels are taken from each sample."""
        labels = torch.tensor([int(s.label) for s in samples], dtype=torch.long)
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= self.num_classes):
            raise ValueError(
                "classification labels must be in "
                f"[0, {self.num_classes}); got min={int(labels.min())}, "
                f"max={int(labels.max())}"
            )
        return pooled, labels

    def evaluate(self, predictions: Tensor, labels: Tensor) -> dict[str, float]:
        return {
            "accuracy": accuracy(predictions, labels),
            "macro_f1": macro_f1(predictions, labels),
        }
