"""Generation task heads and reference-based metrics.

Free-form generation is not part of the phase-1 probing pipeline. This module
exposes a Task implementation that captures the spec and lets the rest of the
pipeline reason about it; metric computation is intentionally minimal until
phase 2.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .base import Task, TaskSpec
from .compatibility import TaskType
from .metrics import exact_match, perplexity

if TYPE_CHECKING:
    from ..datasets.base import DatasetSample


class GenerationTask(Task):
    """Free-form generation evaluated with perplexity as the primary metric.

    Perplexity is reported as ``-log_perplexity`` so that higher is better,
    keeping the convention shared across other task types.
    """

    def __init__(self) -> None:
        self.spec = TaskSpec(
            task_type=TaskType.GENERATION,
            pooling_default="last_token",
            pooling_options=["last_token", "mean"],
            metric_primary="neg_log_perplexity",
            metrics_secondary=["exact_match"],
            allowed_families=["decoder", "encoder_decoder", "ssm"],
            probe_default="linear",
            probe_types_allowed=["linear"],
            needs_labels_for_geometry=False,
        )

    def build_features(
        self,
        samples: list[DatasetSample],
        pooled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self._references = [str(s.label) if s.label is not None else "" for s in samples]
        return pooled, torch.arange(len(samples), dtype=torch.long)

    def evaluate(self, predictions: Tensor, labels: Tensor) -> dict[str, float]:
        """Expect ``predictions`` to be either per-token log-probs or strings."""
        del labels
        if isinstance(predictions, Tensor) and predictions.dtype.is_floating_point:
            ppl = perplexity(predictions)
            return {"neg_log_perplexity": -math.log(ppl), "exact_match": 0.0}
        if isinstance(predictions, Tensor):
            preds_list = [str(p) for p in predictions.tolist()]
        else:
            preds_list = list(predictions)
        em = exact_match(preds_list, getattr(self, "_references", []))
        return {"neg_log_perplexity": 0.0, "exact_match": em}
