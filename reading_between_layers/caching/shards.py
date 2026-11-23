"""On-disk layout for the cache: directory and path resolution given a key."""

from __future__ import annotations

from pathlib import Path

from ..utils.hashing import stable_hash
from .keys import EmbeddingKey

EMBEDDINGS_DIR = "embeddings"
GEOMETRY_DIR = "geometry"
PROBES_DIR = "probes"
SELECTION_DIR = "layer_selection"
MANIFEST_NAME = "manifest.json"


def model_shard(key: EmbeddingKey) -> str:
    """Return the directory name for the model side of an embedding shard."""
    return stable_hash(key.model_id)


def dataset_shard(key: EmbeddingKey) -> str:
    """Return the directory name for the dataset side of an embedding shard."""
    return stable_hash([key.dataset_id, key.dataset_subset])


def embedding_path(root: Path, key: EmbeddingKey) -> Path:
    """Resolve the on-disk path for one cached embedding tensor."""
    return (
        root
        / EMBEDDINGS_DIR
        / model_shard(key)
        / dataset_shard(key)
        / key.split
        / key.pooling_name
        / key.hash()
        / f"layer_{key.layer_index}.pt"
    )


def geometry_dir(root: Path) -> Path:
    """Partition directory under which geometry parquet rows are appended."""
    return root / GEOMETRY_DIR


def probes_dir(root: Path) -> Path:
    """Partition directory under which probe parquet rows are appended."""
    return root / PROBES_DIR


def selection_dir(root: Path) -> Path:
    """Partition directory under which layer-selection rows are appended."""
    return root / SELECTION_DIR


def manifest_path(root: Path) -> Path:
    """File path of the audit manifest."""
    return root / MANIFEST_NAME
