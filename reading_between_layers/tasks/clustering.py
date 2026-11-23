"""Clustering task and label-free evaluation metrics.

Clustering is unusual among our task types: there is no probe to train. We
fit ``KMeans`` on the pooled embeddings layer by layer and compare assignments
against the gold labels using V-measure (primary), NMI, and ARI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from .base import Task, TaskSpec
from .compatibility import TaskType
from .metrics import adjusted_rand_index, normalized_mutual_info, v_measure

if TYPE_CHECKING:
    from ..datasets.base import DatasetSample


class ClusteringTask(Task):
    """KMeans-based clustering evaluation, V-measure primary."""

    def __init__(self, num_clusters: int) -> None:
        self.num_clusters = num_clusters
        self.spec = TaskSpec(
            task_type=TaskType.CLUSTERING,
            pooling_default="mean",
            pooling_options=["mean", "cls"],
            metric_primary="v_measure",
            metrics_secondary=["nmi", "ari"],
            allowed_families=["encoder", "decoder", "encoder_decoder", "ssm"],
            probe_default="linear",
            probe_types_allowed=["linear"],
            needs_labels_for_geometry=True,
        )

    def build_features(
        self,
        samples: list[DatasetSample],
        pooled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        raw_labels = [s.label for s in samples]
        label_to_id = {
            label: idx for idx, label in enumerate(sorted({str(label) for label in raw_labels}))
        }
        labels = torch.tensor(
            [label_to_id[str(label)] for label in raw_labels],
            dtype=torch.long,
        )
        return pooled, labels

    def evaluate(self, predictions: Tensor, labels: Tensor) -> dict[str, float]:
        """``predictions`` are cluster assignments produced by :meth:`cluster`."""
        return {
            "v_measure": v_measure(predictions, labels),
            "nmi": normalized_mutual_info(predictions, labels),
            "ari": adjusted_rand_index(predictions, labels),
        }

    def cluster(self, embeddings: Tensor, seed: int = 42) -> Tensor:
        """Run KMeans with ``num_clusters`` and return integer assignments."""
        from sklearn.cluster import KMeans

        x = embeddings.detach().cpu().float().numpy()
        km = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=seed)
        labels = km.fit_predict(x)
        return torch.from_numpy(np.asarray(labels, dtype=np.int64))
