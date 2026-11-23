"""Parquet-backed append-by-partition store for tabular cache outputs."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from ..constants import PARQUET_COMPRESSION
from ..utils.io import atomic_write_path

_PARQUET_GLOB = "*.parquet"
_SIDECAR_NAMES = {"geometry.parquet"}


def _part_files(partition_dir: Path) -> list[Path]:
    """Return parquet files owned by this append store, excluding sidecars.

    New writes use ``part-*`` names, but older caches used bare UUID parquet
    files. Keep reading both so cache upgrades do not invalidate prior runs.
    """
    return sorted(
        path for path in partition_dir.glob(_PARQUET_GLOB) if path.name not in _SIDECAR_NAMES
    )


def append_row(partition_dir: Path, row: dict[str, Any]) -> Path:
    """Append a single row to ``partition_dir`` as a new parquet file.

    Each call writes one file. Reads concatenate every file in the directory.
    Returns the path of the written file.
    """
    return append_rows(partition_dir, [row])


def append_rows(partition_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Append ``rows`` to ``partition_dir`` as one parquet file."""
    partition_dir.mkdir(parents=True, exist_ok=True)
    out = partition_dir / f"part-{uuid.uuid4().hex}.parquet"
    df = pd.DataFrame(rows)
    with atomic_write_path(out) as tmp:
        df.to_parquet(tmp, compression=PARQUET_COMPRESSION, index=False)
    return out


def read_all(partition_dir: Path) -> pd.DataFrame:
    """Read every parquet file in ``partition_dir`` and concatenate them."""
    if not partition_dir.exists():
        return pd.DataFrame()
    files = _part_files(partition_dir)
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def compact(partition_dir: Path) -> Path | None:
    """Merge every parquet file in ``partition_dir`` into a single file.

    Returns the path of the compacted file, or ``None`` if the directory was empty.
    The pre-existing files are removed only after the compacted file is committed.
    """
    df = read_all(partition_dir)
    if df.empty:
        return None
    out = partition_dir / f"compacted-{uuid.uuid4().hex}.parquet"
    with atomic_write_path(out) as tmp:
        df.to_parquet(tmp, compression=PARQUET_COMPRESSION, index=False)
    for f in _part_files(partition_dir):
        if f != out:
            f.unlink()
    return out
