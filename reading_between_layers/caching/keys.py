"""Cache key dataclasses for embeddings, geometry, and probe outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..utils.hashing import stable_hash


@dataclass(frozen=True)
class EmbeddingKey:
    """Content-addressed key for a per-layer pooled embedding tensor."""

    model_id: str
    dataset_id: str
    dataset_subset: str | None
    split: str
    layer_index: int
    pooling_name: str
    dtype: str
    extraction_max_length: int
    truncation: bool
    sample_order: str = "dataset"
    representation_side: str | None = None

    def hash_payload(self) -> dict[str, Any]:
        """Return the canonical payload used for content addressing."""
        payload = asdict(self)
        if payload.get("representation_side") is None:
            payload.pop("representation_side", None)
        return payload

    def hash(self) -> str:
        """Return the 16-char blake2b hash of the canonical JSON form."""
        return stable_hash(self.hash_payload())


@dataclass(frozen=True)
class GeometryKey:
    """Key for one geometry-metric value over a specific embedding."""

    embedding_key: EmbeddingKey
    metric_name: str
    metric_params_hash: str

    def hash(self) -> str:
        """Return the 16-char blake2b hash of the canonical JSON form."""
        payload = asdict(self)
        payload["embedding_key"] = self.embedding_key.hash_payload()
        return stable_hash(payload)


@dataclass(frozen=True)
class ProbeKey:
    """Key for one trained probe's results on a specific embedding."""

    embedding_key: EmbeddingKey
    task_name: str
    probe_type: str
    seed: int
    probe_config_hash: str

    def hash(self) -> str:
        """Return the 16-char blake2b hash of the canonical JSON form."""
        payload = asdict(self)
        payload["embedding_key"] = self.embedding_key.hash_payload()
        return stable_hash(payload)
