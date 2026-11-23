"""Content-addressed cache front-end for embeddings, geometry, and probe results."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from torch import Tensor

from ..exceptions import CacheMiss
from ..utils.io import atomic_write_path, load_tensor, save_tensor
from .keys import EmbeddingKey, GeometryKey, ProbeKey
from .parquet_store import append_row, append_rows, read_all
from .shards import (
    embedding_path,
    geometry_dir,
    manifest_path,
    probes_dir,
)

logger = logging.getLogger(__name__)

_KIND_EMBEDDING = "embedding"
_KIND_GEOMETRY = "geometry"
_KIND_PROBE = "probe"


class CacheStore:
    """Content-addressed cache for embeddings, geometry, and probe results.

    All paths are resolved relative to ``root``. Embeddings are stored as one
    torch ``.pt`` file per key. Geometry and probe rows are appended as parquet
    files under their respective partition directories. Every successful write
    appends a line to ``manifest.json`` for audit purposes.
    """

    def __init__(self, root: Path, verify_on_read: bool = False) -> None:
        self.root = Path(root)
        self.verify_on_read = verify_on_read
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_hashes: dict[str, set[str]] | None = None
        self._table_cache: dict[str, pd.DataFrame] = {}

    def has_embedding(self, key: EmbeddingKey) -> bool:
        """Return True iff the embedding for ``key`` is present in the cache."""
        return embedding_path(self.root, key).exists()

    def load_embedding(self, key: EmbeddingKey) -> Tensor:
        """Load the cached embedding tensor for ``key``.

        Raises:
            CacheMiss: if no embedding is cached for this key.
        """
        path = embedding_path(self.root, key)
        if not path.exists():
            raise CacheMiss(f"no embedding cached at {path}")
        return load_tensor(path)

    def load_embedding_mmap(self, key: EmbeddingKey) -> Tensor:
        """Memory-map the cached embedding tensor for batch-wise probe reads."""
        path = embedding_path(self.root, key)
        if not path.exists():
            raise CacheMiss(f"no embedding cached at {path}")
        return load_tensor(path, mmap=True)

    def save_embedding(self, key: EmbeddingKey, tensor: Tensor) -> None:
        """Atomically save ``tensor`` for ``key`` and record it in the manifest."""
        path = embedding_path(self.root, key)
        save_tensor(tensor, path)
        self._record(_KIND_EMBEDDING, key.hash(), asdict(key))

    def has_geometry(self, key: GeometryKey) -> bool:
        """Return True iff a geometry row is present for ``key``."""
        return self._has_record(_KIND_GEOMETRY, key.hash(), geometry_dir(self.root))

    def load_geometry(self, key: GeometryKey) -> dict[str, Any]:
        """Load the geometry value dict cached for ``key``.

        Raises:
            CacheMiss: if no row is cached for this key.
        """
        df = self._read_table(_KIND_GEOMETRY, geometry_dir(self.root))
        if df.empty or "hash" not in df.columns:
            raise CacheMiss(f"no geometry cached for {key.hash()}")
        rows = df[df["hash"] == key.hash()]
        if rows.empty:
            raise CacheMiss(f"no geometry cached for {key.hash()}")
        return json.loads(rows.iloc[-1]["value_json"])

    def save_geometry(self, key: GeometryKey, value: dict[str, Any]) -> None:
        """Append a geometry row for ``key`` and record it in the manifest."""
        self.save_geometry_many([(key, value)])

    def save_geometry_many(self, items: list[tuple[GeometryKey, dict[str, Any]]]) -> None:
        """Append several geometry rows in one parquet part and manifest write."""
        if not items:
            return
        rows: list[dict[str, Any]] = []
        manifest_rows: list[tuple[str, str, dict[str, Any]]] = []
        for key, value in items:
            row = {
                "hash": key.hash(),
                "metric_name": key.metric_name,
                "metric_params_hash": key.metric_params_hash,
                "embedding_hash": key.embedding_key.hash(),
                "value_json": json.dumps(value),
            }
            rows.append(row)
            manifest_rows.append((_KIND_GEOMETRY, key.hash(), asdict(key)))
        append_rows(geometry_dir(self.root), rows)
        self._append_cached_rows(_KIND_GEOMETRY, rows)
        self._record_many(manifest_rows)

    def _legacy_save_geometry(self, key: GeometryKey, value: dict[str, Any]) -> None:
        """Previous single-row implementation, kept as reference for debugging."""
        row = {
            "hash": key.hash(),
            "metric_name": key.metric_name,
            "metric_params_hash": key.metric_params_hash,
            "embedding_hash": key.embedding_key.hash(),
            "value_json": json.dumps(value),
        }
        append_row(geometry_dir(self.root), row)
        self._append_cached_row(_KIND_GEOMETRY, row)
        self._record(_KIND_GEOMETRY, key.hash(), asdict(key))

    def has_probe(self, key: ProbeKey) -> bool:
        """Return True iff a probe row is present for ``key``."""
        return self._has_record(_KIND_PROBE, key.hash(), probes_dir(self.root))

    def load_probe(self, key: ProbeKey) -> dict[str, Any]:
        """Load the probe-result dict cached for ``key``.

        Raises:
            CacheMiss: if no row is cached for this key.
        """
        df = self._read_table(_KIND_PROBE, probes_dir(self.root))
        if df.empty or "hash" not in df.columns:
            raise CacheMiss(f"no probe cached for {key.hash()}")
        rows = df[df["hash"] == key.hash()]
        if rows.empty:
            raise CacheMiss(f"no probe cached for {key.hash()}")
        return json.loads(rows.iloc[-1]["value_json"])

    def save_probe(self, key: ProbeKey, value: dict[str, Any]) -> None:
        """Append a probe row for ``key`` and record it in the manifest."""
        self.save_probe_many([(key, value)])

    def save_probe_many(self, items: list[tuple[ProbeKey, dict[str, Any]]]) -> None:
        """Append several probe rows in one parquet part and manifest write."""
        if not items:
            return
        rows: list[dict[str, Any]] = []
        manifest_rows: list[tuple[str, str, dict[str, Any]]] = []
        for key, value in items:
            row = {
                "hash": key.hash(),
                "task_name": key.task_name,
                "probe_type": key.probe_type,
                "seed": key.seed,
                "probe_config_hash": key.probe_config_hash,
                "embedding_hash": key.embedding_key.hash(),
                "value_json": json.dumps(value),
            }
            rows.append(row)
            manifest_rows.append((_KIND_PROBE, key.hash(), asdict(key)))
        append_rows(probes_dir(self.root), rows)
        self._append_cached_rows(_KIND_PROBE, rows)
        self._record_many(manifest_rows)

    def _legacy_save_probe(self, key: ProbeKey, value: dict[str, Any]) -> None:
        """Previous single-row implementation, kept as reference for debugging."""
        row = {
            "hash": key.hash(),
            "task_name": key.task_name,
            "probe_type": key.probe_type,
            "seed": key.seed,
            "probe_config_hash": key.probe_config_hash,
            "embedding_hash": key.embedding_key.hash(),
            "value_json": json.dumps(value),
        }
        append_row(probes_dir(self.root), row)
        self._append_cached_row(_KIND_PROBE, row)
        self._record(_KIND_PROBE, key.hash(), asdict(key))

    def get_or_compute_embedding(
        self,
        key: EmbeddingKey,
        compute_fn: Callable[[], Tensor],
    ) -> Tensor:
        """Return the cached embedding, computing and caching it on miss."""
        if self.has_embedding(key):
            return self.load_embedding(key)
        tensor = compute_fn()
        self.save_embedding(key, tensor)
        return tensor

    def get_or_compute_geometry(
        self,
        key: GeometryKey,
        compute_fn: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Return the cached geometry value, computing and caching it on miss."""
        if self.has_geometry(key):
            return self.load_geometry(key)
        value = compute_fn()
        self.save_geometry(key, value)
        return value

    def get_or_compute_probe(
        self,
        key: ProbeKey,
        compute_fn: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Return the cached probe result, computing and caching it on miss."""
        if self.has_probe(key):
            return self.load_probe(key)
        value = compute_fn()
        self.save_probe(key, value)
        return value

    def manifest(self) -> pd.DataFrame:
        """Return the audit manifest as a DataFrame.

        Each row corresponds to one ``save_*`` call. Columns include ``kind`` and
        ``hash``; the full key payload is in ``key_json``.
        """
        path = manifest_path(self.root)
        if not path.exists():
            return pd.DataFrame(columns=["kind", "hash", "key_json"])
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if not rows:
            return pd.DataFrame(columns=["kind", "hash", "key_json"])
        return pd.DataFrame(rows)

    def purge(self, predicate: Callable[[dict[str, Any]], bool]) -> int:
        """Remove embedding files whose manifest entry satisfies ``predicate``.

        Returns the number of files removed. Tabular rows in geometry/probes
        parquet partitions are not modified by this method.
        """
        df = self.manifest()
        if df.empty:
            return 0
        removed = 0
        kept: list[dict[str, Any]] = []
        for record in df.to_dict(orient="records"):
            if predicate(record):
                if record.get("kind") == _KIND_EMBEDDING:
                    key_payload = json.loads(record["key_json"])
                    key = EmbeddingKey(**key_payload)
                    path = embedding_path(self.root, key)
                    if path.exists():
                        path.unlink()
                        removed += 1
            else:
                kept.append(record)
        self._rewrite_manifest(kept)
        return removed

    def _record(self, kind: str, key_hash: str, key_payload: dict[str, Any]) -> None:
        """Append one entry to the JSONL audit manifest."""
        path = manifest_path(self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"kind": kind, "hash": key_hash, "key_json": json.dumps(key_payload)}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        if self._manifest_hashes is not None:
            self._manifest_hashes.setdefault(kind, set()).add(key_hash)

    def _record_many(self, rows: list[tuple[str, str, dict[str, Any]]]) -> None:
        """Append several JSONL audit manifest entries with one file open."""
        if not rows:
            return
        path = manifest_path(self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = [
            {"kind": kind, "hash": key_hash, "key_json": json.dumps(key_payload)}
            for kind, key_hash, key_payload in rows
        ]
        with path.open("a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        if self._manifest_hashes is not None:
            for kind, key_hash, _ in rows:
                self._manifest_hashes.setdefault(kind, set()).add(key_hash)

    def _rewrite_manifest(self, rows: list[dict[str, Any]]) -> None:
        """Replace the manifest file atomically with the given rows."""
        path = manifest_path(self.root)
        with atomic_write_path(path) as tmp:
            with tmp.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        self._manifest_hashes = None

    def _manifest_hash_index(self) -> dict[str, set[str]]:
        """Return a cached mapping from manifest kind to content hashes."""
        if self._manifest_hashes is not None:
            return self._manifest_hashes
        path = manifest_path(self.root)
        hashes: dict[str, set[str]] = {
            _KIND_EMBEDDING: set(),
            _KIND_GEOMETRY: set(),
            _KIND_PROBE: set(),
        }
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("skipping unparseable manifest line")
                        continue
                    kind = record.get("kind")
                    key_hash = record.get("hash")
                    if isinstance(kind, str) and isinstance(key_hash, str):
                        hashes.setdefault(kind, set()).add(key_hash)
        self._manifest_hashes = hashes
        return hashes

    def _has_record(self, kind: str, key_hash: str, partition_dir: Path) -> bool:
        """Fast existence check backed by manifest hashes, with legacy fallback."""
        manifest = manifest_path(self.root)
        if manifest.exists():
            return key_hash in self._manifest_hash_index().get(kind, set())
        df = self._read_table(kind, partition_dir)
        if df.empty or "hash" not in df.columns:
            return False
        return bool((df["hash"] == key_hash).any())

    def _read_table(self, kind: str, partition_dir: Path) -> pd.DataFrame:
        """Read and cache a parquet partition table for this CacheStore process."""
        cached = self._table_cache.get(kind)
        if cached is not None:
            return cached
        df = read_all(partition_dir)
        self._table_cache[kind] = df
        return df

    def _append_cached_row(self, kind: str, row: dict[str, Any]) -> None:
        """Keep an already-loaded table cache coherent after appending a row."""
        self._append_cached_rows(kind, [row])

    def _append_cached_rows(self, kind: str, rows: list[dict[str, Any]]) -> None:
        """Keep an already-loaded table cache coherent after appending rows."""
        if kind not in self._table_cache:
            return
        appended = pd.DataFrame(rows)
        current = self._table_cache[kind]
        self._table_cache[kind] = pd.concat([current, appended], ignore_index=True)
