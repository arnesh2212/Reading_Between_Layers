"""Multiple-choice QA task with prompt-level or per-option scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .base import Task, TaskSpec
from .compatibility import TaskType
from .metrics import accuracy

if TYPE_CHECKING:
    from ..datasets.base import DatasetSample


class MultipleChoiceQATask(Task):
    """Multiple-choice QA with a shared scalar option-scoring head."""

    def __init__(self, num_options: int) -> None:
        self.num_options = num_options
        self.spec = TaskSpec(
            task_type=TaskType.MULTIPLE_CHOICE_QA,
            pooling_default="mean",
            pooling_options=["mean", "last_token", "cls"],
            metric_primary="accuracy",
            metrics_secondary=[],
            allowed_families=["encoder", "decoder", "encoder_decoder", "ssm"],
            probe_default="linear",
            probe_types_allowed=["linear", "nonlinear"],
            needs_labels_for_geometry=False,
        )

    def build_features(
        self,
        samples: list[DatasetSample],
        pooled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build labels for prompt-level or flattened per-option embeddings.

        The local sweep uses one prompt per question with all answer choices in
        the text, producing ``(n, hidden)`` embeddings. We also keep support for
        older flattened option rows, ``(n * num_options, hidden)``.
        """
        n = len(samples)
        labels = torch.tensor([int(s.label) for s in samples], dtype=torch.long)
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= self.num_options):
            raise ValueError(
                "multiple-choice labels must be in "
                f"[0, {self.num_options}); got min={int(labels.min())}, "
                f"max={int(labels.max())}"
            )
        if pooled.size(0) == n:
            self._option_mask = None
            return pooled, labels
        if pooled.size(0) != n * self.num_options:
            raise ValueError(
                f"expected {n} or {n * self.num_options} pooled rows, got {pooled.size(0)}"
            )
        reshaped = pooled.view(n, self.num_options, -1)
        masks = []
        any_mask = False
        for s in samples:
            mask = s.extras.get("option_mask")
            if mask is None:
                masks.append([True] * self.num_options)
            else:
                any_mask = True
                masks.append([bool(m) for m in mask])
        self._option_mask = torch.tensor(masks, dtype=torch.bool) if any_mask else None
        return reshaped, labels

    def evaluate(self, predictions: Tensor, labels: Tensor) -> dict[str, float]:
        return {"accuracy": accuracy(predictions, labels)}
