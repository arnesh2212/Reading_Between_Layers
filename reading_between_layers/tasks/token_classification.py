"""Token classification task: per-token head, BIO-aware metrics via seqeval."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .base import Task, TaskSpec
from .compatibility import TaskType
from .metrics import token_classification_f1

if TYPE_CHECKING:
    from ..datasets.base import DatasetSample


class TokenClassificationTask(Task):
    """Token-level classification (NER, POS) evaluated with seqeval F1.

    ``build_features`` expects ``pooled`` to actually be token-level
    embeddings of shape ``(total_tokens, hidden)``; the upstream extractor
    skips pooling and writes the per-token tensor. ``samples`` provide
    per-example BIO tag sequences; we flatten them to match the rows of
    ``pooled``.
    """

    def __init__(self, label_names: list[str]) -> None:
        self.label_names = label_names
        self.num_classes = len(label_names)
        self._id_to_label = dict(enumerate(label_names))
        self.spec = TaskSpec(
            task_type=TaskType.TOKEN_CLASSIFICATION,
            pooling_default="none",
            pooling_options=["none"],
            metric_primary="f1",
            metrics_secondary=["accuracy"],
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
        flat_labels: list[int] = []
        offsets: list[int] = []
        for sample in samples:
            tags = sample.label
            if tags is None:
                tags = []
            offsets.append(len(flat_labels))
            flat_labels.extend(int(t) for t in tags)
        if pooled.size(0) != len(flat_labels):
            raise ValueError(
                f"token features have {pooled.size(0)} rows but labels have {len(flat_labels)}"
            )
        self._offsets = offsets
        self._sample_lengths = [
            (offsets[i + 1] - offsets[i])
            if i + 1 < len(offsets)
            else len(flat_labels) - offsets[i]
            for i in range(len(offsets))
        ]
        return pooled, torch.tensor(flat_labels, dtype=torch.long)

    def evaluate(self, predictions: Tensor, labels: Tensor) -> dict[str, float]:
        offsets = getattr(self, "_offsets", None)
        lengths = getattr(self, "_sample_lengths", None)
        if offsets is None or lengths is None:
            offsets = [0]
            lengths = [int(labels.numel())]
        preds_per_seq: list[list[str]] = []
        labels_per_seq: list[list[str]] = []
        preds_flat = predictions.tolist()
        labels_flat = labels.tolist()
        for start, length in zip(offsets, lengths, strict=True):
            preds_per_seq.append(
                [self._id_to_label[int(p)] for p in preds_flat[start : start + length]]
            )
            labels_per_seq.append(
                [self._id_to_label[int(y)] for y in labels_flat[start : start + length]]
            )
        try:
            f1 = token_classification_f1(preds_per_seq, labels_per_seq)
        except ImportError:
            f1 = float("nan")
        flat_acc = float((predictions == labels).float().mean()) if labels.numel() else 0.0
        return {"f1": f1, "accuracy": flat_acc}
