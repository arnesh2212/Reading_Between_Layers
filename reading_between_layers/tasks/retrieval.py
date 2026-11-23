"""Retrieval task: bi-encoder scoring, MRR, recall, and nDCG.

Phase 1 uses an in-batch contrastive evaluation: for a batch of (query,
positive) pairs we treat the positives of all other queries as in-batch
negatives. The evaluation matrix is ``(n, n)`` cosine similarities with the
diagonal as the ground-truth positive. Full BEIR evaluation (separate corpus,
hard negatives) is deferred to a later phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .base import Task, TaskSpec
from .compatibility import TaskType
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k

if TYPE_CHECKING:
    from ..datasets.base import DatasetSample


class RetrievalTask(Task):
    """Bi-encoder retrieval evaluated with in-batch contrastive scoring."""

    def __init__(self, subset_size: int = 256, k_recall: int = 100, k_ranking: int = 10) -> None:
        self.subset_size = subset_size
        self.k_recall = k_recall
        self.k_ranking = k_ranking
        self.spec = TaskSpec(
            task_type=TaskType.RETRIEVAL,
            pooling_default="mean",
            pooling_options=["mean", "cls"],
            metric_primary="mrr_at_10",
            metrics_secondary=["ndcg_at_10", "recall_at_100"],
            allowed_families=["encoder", "encoder_decoder", "ssm"],
            probe_default="linear",
            probe_types_allowed=["linear"],
            needs_labels_for_geometry=False,
        )

    def build_features(
        self,
        samples: list[DatasetSample],
        pooled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(pooled, indices)`` where indices are the diagonal labels."""
        del samples
        n = pooled.size(0)
        return pooled, torch.arange(n, dtype=torch.long)

    def evaluate(self, predictions: Tensor, labels: Tensor) -> dict[str, float]:
        """``predictions`` is the score matrix ``(n, n)``; ``labels`` is unused."""
        del labels
        scores = predictions
        return {
            "mrr_at_10": mrr_at_k(scores, k=self.k_ranking),
            "ndcg_at_10": ndcg_at_k(scores, k=self.k_ranking),
            "recall_at_100": recall_at_k(scores, k=self.k_recall),
        }

    def score_matrix(self, embeddings: Tensor) -> Tensor:
        """Cosine-similarity score matrix used at evaluation time."""
        x = embeddings.float()
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
        return x @ x.t()
