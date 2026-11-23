"""Tests for the content-addressed cache layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from reading_between_layers.caching.keys import EmbeddingKey, GeometryKey, ProbeKey
from reading_between_layers.caching.parquet_store import append_row, compact, read_all
from reading_between_layers.caching.shards import embedding_path
from reading_between_layers.caching.store import CacheStore
from reading_between_layers.exceptions import CacheMiss


def _embedding_key(layer: int = 0, model: str = "m", dataset: str = "d") -> EmbeddingKey:
    return EmbeddingKey(
        model_id=model,
        dataset_id=dataset,
        dataset_subset=None,
        split="train",
        layer_index=layer,
        pooling_name="mean",
        dtype="float16",
        extraction_max_length=512,
        truncation=True,
    )


def test_embedding_key_hash_is_stable_and_16_chars():
    k1 = _embedding_key()
    k2 = _embedding_key()
    assert k1.hash() == k2.hash()
    assert len(k1.hash()) == 16
    assert all(c in "0123456789abcdef" for c in k1.hash())


def test_different_embedding_keys_yield_different_hashes():
    k1 = _embedding_key(layer=0)
    k2 = _embedding_key(layer=1)
    k3 = _embedding_key(model="other")
    assert k1.hash() != k2.hash()
    assert k1.hash() != k3.hash()
    assert k2.hash() != k3.hash()


def test_representation_side_only_changes_hash_when_set():
    legacy = _embedding_key()
    explicit_default = EmbeddingKey(**legacy.hash_payload(), representation_side=None)
    decoder_side = EmbeddingKey(**legacy.hash_payload(), representation_side="decoder")

    assert explicit_default.hash() == legacy.hash()
    assert decoder_side.hash() != legacy.hash()


def test_geometry_and_probe_keys_hash_with_nested_embedding():
    ek = _embedding_key()
    g1 = GeometryKey(embedding_key=ek, metric_name="isotropy", metric_params_hash="abc")
    g2 = GeometryKey(embedding_key=ek, metric_name="rankme", metric_params_hash="abc")
    assert g1.hash() != g2.hash()

    p1 = ProbeKey(
        embedding_key=ek,
        task_name="sst2",
        probe_type="linear",
        seed=42,
        probe_config_hash="cfg",
    )
    p2 = ProbeKey(
        embedding_key=ek,
        task_name="sst2",
        probe_type="linear",
        seed=1337,
        probe_config_hash="cfg",
    )
    assert p1.hash() != p2.hash()


def test_embedding_round_trip(tmp_path: Path):
    store = CacheStore(tmp_path)
    key = _embedding_key()
    tensor = torch.randn(8, 16).to(torch.float16)
    store.save_embedding(key, tensor)

    loaded = store.load_embedding(key)
    assert loaded.shape == tensor.shape
    assert loaded.dtype == tensor.dtype
    assert torch.equal(loaded, tensor)


def test_embedding_mmap_round_trip_uses_mmap_load(tmp_path: Path):
    store = CacheStore(tmp_path)
    key = _embedding_key()
    tensor = torch.randn(8, 16).to(torch.float16)
    store.save_embedding(key, tensor)

    with patch("reading_between_layers.utils.io.torch.load", wraps=torch.load) as mocked_load:
        loaded = store.load_embedding_mmap(key)

    assert torch.equal(loaded, tensor)
    assert mocked_load.call_args.kwargs["mmap"] is True


def test_get_or_compute_embedding_calls_compute_once(tmp_path: Path):
    store = CacheStore(tmp_path)
    key = _embedding_key()
    counter = {"n": 0}

    def compute() -> torch.Tensor:
        counter["n"] += 1
        return torch.zeros(4, 4, dtype=torch.float16)

    first = store.get_or_compute_embedding(key, compute)
    second = store.get_or_compute_embedding(key, compute)

    assert counter["n"] == 1
    assert torch.equal(first, second)


def test_load_embedding_raises_cache_miss(tmp_path: Path):
    store = CacheStore(tmp_path)
    key = _embedding_key()
    with pytest.raises(CacheMiss):
        store.load_embedding(key)


def test_atomic_write_leaves_no_partial_file(tmp_path: Path):
    store = CacheStore(tmp_path)
    key = _embedding_key()
    tensor = torch.zeros(4, 4, dtype=torch.float16)

    with patch("reading_between_layers.utils.io.torch.save", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            store.save_embedding(key, tensor)

    expected = embedding_path(tmp_path, key)
    assert not expected.exists()
    leftovers = list(tmp_path.rglob("*.pt")) + list(tmp_path.rglob("*.tmp"))
    assert leftovers == []


def test_parquet_append_then_read_all_rows(tmp_path: Path):
    partition = tmp_path / "partition"
    for i in range(10):
        append_row(partition, {"i": i, "val": i * 2})
    df = read_all(partition)
    assert len(df) == 10
    assert sorted(df["i"].tolist()) == list(range(10))
    assert sorted(df["val"].tolist()) == [i * 2 for i in range(10)]


def test_parquet_compact_merges_files(tmp_path: Path):
    partition = tmp_path / "partition"
    for i in range(5):
        append_row(partition, {"i": i})
    compacted = compact(partition)
    assert compacted is not None
    parquet_files = list(partition.glob("*.parquet"))
    assert parquet_files == [compacted]
    df = read_all(partition)
    assert len(df) == 5


def test_manifest_lists_every_key_written(tmp_path: Path):
    store = CacheStore(tmp_path)
    k1 = _embedding_key(layer=0)
    k2 = _embedding_key(layer=1)
    store.save_embedding(k1, torch.zeros(2, 2, dtype=torch.float16))
    store.save_embedding(k2, torch.zeros(2, 2, dtype=torch.float16))

    gk = GeometryKey(embedding_key=k1, metric_name="isotropy", metric_params_hash="x")
    store.save_geometry(gk, {"value": 0.5})

    manifest = store.manifest()
    hashes = set(manifest["hash"].tolist())
    assert k1.hash() in hashes
    assert k2.hash() in hashes
    assert gk.hash() in hashes
    kinds = set(manifest["kind"].tolist())
    assert kinds == {"embedding", "geometry"}


def test_geometry_round_trip(tmp_path: Path):
    store = CacheStore(tmp_path)
    ek = _embedding_key()
    gk = GeometryKey(embedding_key=ek, metric_name="rankme", metric_params_hash="p")
    assert not store.has_geometry(gk)
    store.save_geometry(gk, {"rankme": 12.3, "n": 100})
    assert store.has_geometry(gk)
    assert store.load_geometry(gk) == {"rankme": 12.3, "n": 100}


def test_geometry_has_uses_manifest_without_scanning_parquet(tmp_path: Path):
    store = CacheStore(tmp_path)
    ek = _embedding_key()
    gk = GeometryKey(embedding_key=ek, metric_name="rankme", metric_params_hash="p")
    store.save_geometry(gk, {"rankme": 12.3})

    fresh_store = CacheStore(tmp_path)
    with patch("reading_between_layers.caching.store.read_all", side_effect=AssertionError("slow scan")):
        assert fresh_store.has_geometry(gk)


def test_probe_round_trip(tmp_path: Path):
    store = CacheStore(tmp_path)
    ek = _embedding_key()
    pk = ProbeKey(
        embedding_key=ek,
        task_name="sst2",
        probe_type="linear",
        seed=42,
        probe_config_hash="cfg",
    )
    assert not store.has_probe(pk)
    store.save_probe(pk, {"accuracy": 0.91})
    assert store.has_probe(pk)
    assert store.load_probe(pk) == {"accuracy": 0.91}


def test_probe_has_uses_manifest_without_scanning_parquet(tmp_path: Path):
    store = CacheStore(tmp_path)
    ek = _embedding_key()
    pk = ProbeKey(
        embedding_key=ek,
        task_name="sst2",
        probe_type="linear",
        seed=42,
        probe_config_hash="cfg",
    )
    store.save_probe(pk, {"accuracy": 0.91})

    fresh_store = CacheStore(tmp_path)
    with patch("reading_between_layers.caching.store.read_all", side_effect=AssertionError("slow scan")):
        assert fresh_store.has_probe(pk)
