"""Abstract Task interface plus the TaskSpec dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from torch import Tensor

from .compatibility import TaskType

if TYPE_CHECKING:
    from ..datasets.base import DatasetSample


@dataclass
class TaskSpec:
    """Static description of a task type from the perspective of probing.

    Mirrors the on-disk task config (``configs/task/<name>.yaml``). Used by
    :class:`Task` subclasses to expose their metric names and allowed pooling /
    probe combinations to the rest of the pipeline.
    """

    task_type: TaskType
    pooling_default: str
    pooling_options: list[str]
    metric_primary: str
    metrics_secondary: list[str]
    allowed_families: list[str]
    probe_default: str
    probe_types_allowed: list[str] = field(default_factory=list)
    needs_labels_for_geometry: bool = False


class Task(ABC):
    """Abstract base class for one task type.

    Concrete subclasses live in sibling modules (``classification.py``,
    ``token_classification.py``, etc.). Each subclass implements
    :meth:`build_features` to turn pooled embeddings plus dataset samples into a
    ``(X, y)`` pair suitable for the shared probe trainer, and :meth:`evaluate`
    to produce a metric dictionary from raw predictions and labels.
    """

    spec: TaskSpec

    @abstractmethod
    def build_features(
        self,
        samples: list[DatasetSample],
        pooled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Convert pooled embeddings plus samples into ``(X, y)`` for the probe.

        Args:
            samples: list of dataset samples in the same order as ``pooled``.
            pooled: pooled embeddings, shape ``(n, hidden)`` for most tasks, or
                ``(n * num_options, hidden)`` for multiple choice QA.

        Returns:
            ``(X, y)`` where ``X`` is the feature tensor passed to the probe
            head and ``y`` is the label tensor used by the trainer's loss.
        """

    @abstractmethod
    def evaluate(self, predictions: Tensor, labels: Tensor) -> dict[str, float]:
        """Compute a metric dictionary keyed by metric name.

        The first entry is the primary metric (matches ``spec.metric_primary``).
        Higher is better for the primary metric across all task types so that
        downstream model-selection logic does not need a per-task direction
        table.
        """
