"""Extractive and generative QA task heads and metrics.

Both phase-1 implementations are placeholders that wire the task spec and the
metrics; the full extractive head (start/end span over token positions) and
generative answer-likelihood scoring are routed through their dedicated probe
modules in phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .base import Task, TaskSpec
from .compatibility import TaskType
from .metrics import exact_match, qa_token_f1

if TYPE_CHECKING:
    from ..datasets.base import DatasetSample


class ExtractiveQATask(Task):
    """Extractive QA evaluated with token-overlap F1 (primary) and exact match.

    ``build_features`` returns the pooled embedding alongside per-sample answer
    strings stored in ``extras``; the trainer supplies them to the span head.
    """

    def __init__(self) -> None:
        self.spec = TaskSpec(
            task_type=TaskType.QA_EXTRACTIVE,
            pooling_default="mean",
            pooling_options=["mean", "cls"],
            metric_primary="f1",
            metrics_secondary=["exact_match"],
            allowed_families=["encoder", "decoder", "encoder_decoder", "ssm"],
            probe_default="linear",
            probe_types_allowed=["linear"],
            needs_labels_for_geometry=False,
        )

    def build_features(
        self,
        samples: list[DatasetSample],
        pooled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self._gold_answers = [str(s.extras.get("answer_text", s.label or "")) for s in samples]
        return pooled, torch.arange(len(samples), dtype=torch.long)

    def evaluate(self, predictions: Tensor, labels: Tensor) -> dict[str, float]:
        del labels
        if isinstance(predictions, Tensor):
            preds_list = [str(p) for p in predictions.tolist()]
        else:
            preds_list = list(predictions)
        gold = getattr(self, "_gold_answers", [])
        return {
            "f1": qa_token_f1(preds_list, gold),
            "exact_match": exact_match(preds_list, gold),
        }


class GenerativeQATask(Task):
    """Generative QA (open-ended answers) scored by exact match."""

    def __init__(self) -> None:
        self.spec = TaskSpec(
            task_type=TaskType.QA_GENERATIVE,
            pooling_default="last_token",
            pooling_options=["last_token", "mean"],
            metric_primary="accuracy",
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
        self._gold_answers = [str(s.label) if s.label is not None else "" for s in samples]
        return pooled, torch.arange(len(samples), dtype=torch.long)

    def evaluate(self, predictions: Tensor, labels: Tensor) -> dict[str, float]:
        del labels
        if isinstance(predictions, Tensor):
            preds_list = [str(p) for p in predictions.tolist()]
        else:
            preds_list = list(predictions)
        gold = getattr(self, "_gold_answers", [])
        em = exact_match(preds_list, gold)
        return {"accuracy": em, "exact_match": em}
