"""Tests for the geometry pipeline added in prompt 16."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from reading_between_layers.caching.keys import EmbeddingKey, GeometryKey
from reading_between_layers.caching.store import CacheStore
from reading_between_layers.geometry import compute_all_geometry, resolve_geometry_sample_size
from reading_between_layers.utils.hashing import stable_hash


def _embedding_key(layer: int = 1) -> EmbeddingKey:
    return EmbeddingKey(
        model_id="test/model",
        dataset_id="test/dataset",
        dataset_subset=None,
        split="train",
        layer_index=layer,
        pooling_name="cls",
        dtype="float16",
        extraction_max_length=8,
        truncation=True,
    )


def _fast_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "metrics": {
                "effective_rank": True,
                "stable_rank": True,
                "participation_ratio": True,
                "anisotropy_random_pairs": True,
            },
            "sample_size": None,
        }
    )


def test_compute_all_geometry_runs_enabled_metrics():
    torch.manual_seed(0)
    out = compute_all_geometry(torch.randn(200, 32), labels=None, config=_fast_cfg())
    for key in ("effective_rank", "stable_rank", "participation_ratio", "anisotropy_random_pairs"):
        assert key in out
        assert math.isfinite(out[key])


def test_geometry_sample_fraction_resolves_to_ten_percent():
    cfg = OmegaConf.create({"sample_size": None, "sample_fraction": 0.1})
    assert resolve_geometry_sample_size(100, cfg) == 10
    assert resolve_geometry_sample_size(101, cfg) == 10
    assert resolve_geometry_sample_size(3, cfg) == 1


def test_geometry_sample_size_overrides_fraction():
    cfg = OmegaConf.create({"sample_size": 7, "sample_fraction": 0.1})
    assert resolve_geometry_sample_size(100, cfg) == 7


def test_compute_all_geometry_failed_metric_returns_nan(monkeypatch):
    from reading_between_layers.geometry import registry

    class Broken:
        name = "broken"
        requires_labels = False
        requires_jacobian = False
        output_keys = ["broken"]

        def compute(self, embeddings, labels=None, attention_mask=None, model=None):
            raise RuntimeError("intentional")

    patched = {**registry.GEOMETRY_REGISTRY, "broken": Broken}
    monkeypatch.setattr(registry, "GEOMETRY_REGISTRY", patched)
    # The compute_all_geometry import already grabbed a reference to the dict;
    # patch the module-level reference too.
    import reading_between_layers.geometry as geom_pkg

    monkeypatch.setattr(geom_pkg, "GEOMETRY_REGISTRY", patched)

    cfg = OmegaConf.create({"metrics": {"broken": True}, "sample_size": None})
    out = compute_all_geometry(torch.randn(50, 8), labels=None, config=cfg)
    assert math.isnan(out["broken"])


def test_geometry_rows_for_layer_skips_already_cached(tmp_path: Path):
    cache = CacheStore(tmp_path)
    key = _embedding_key()
    cache.save_embedding(key, torch.randn(40, 8))
    cfg = OmegaConf.create(
        {
            "geometry": _fast_cfg(),
            "cache": {"embeddings_dtype": "float16"},
        }
    )
    pipeline_path = (
        Path(__file__).resolve().parent.parent / "pipeline" / "02_compute_geometry.py"
    )
    spec = importlib.util.spec_from_file_location("compute_geometry_pipeline", pipeline_path)
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)

    embedding = cache.load_embedding(key)
    first = pipeline.geometry_rows_for_layer(
        embedding_key=key,
        embedding=embedding,
        labels=None,
        cfg=cfg,
        cache=cache,
        model_id=key.model_id,
        dataset_id=key.dataset_id,
    )
    assert len(first) >= 4
    second = pipeline.geometry_rows_for_layer(
        embedding_key=key,
        embedding=embedding,
        labels=None,
        cfg=cfg,
        cache=cache,
        model_id=key.model_id,
        dataset_id=key.dataset_id,
    )
    assert second == []


def test_geometry_failed_metric_records_nan_row_without_aborting(tmp_path: Path, monkeypatch):
    """A failing metric should produce a NaN row, not crash the pipeline."""
    cache = CacheStore(tmp_path)
    key = _embedding_key()
    cache.save_embedding(key, torch.randn(40, 8))

    pipeline_path = (
        Path(__file__).resolve().parent.parent / "pipeline" / "02_compute_geometry.py"
    )
    spec = importlib.util.spec_from_file_location("compute_geometry_pipeline2", pipeline_path)
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)

    def raising_compute(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(pipeline, "compute_all_geometry", raising_compute)
    cfg = OmegaConf.create(
        {
            "geometry": OmegaConf.create(
                {"metrics": {"effective_rank": True}, "sample_size": None}
            ),
            "cache": {"embeddings_dtype": "float16"},
        }
    )
    embedding = cache.load_embedding(key)
    rows = pipeline.geometry_rows_for_layer(
        embedding_key=key,
        embedding=embedding,
        labels=None,
        cfg=cfg,
        cache=cache,
        model_id=key.model_id,
        dataset_id=key.dataset_id,
    )
    assert rows
    assert math.isnan(rows[0]["metric_value"])
    assert rows[0]["failure_reason"]


def test_cached_geometry_skips_jacobian_metric_without_live_model(tmp_path: Path, caplog):
    cache = CacheStore(tmp_path)
    key = _embedding_key()
    cache.save_embedding(key, torch.randn(40, 8))

    pipeline_path = (
        Path(__file__).resolve().parent.parent / "pipeline" / "02_compute_geometry.py"
    )
    spec = importlib.util.spec_from_file_location("compute_geometry_pipeline3", pipeline_path)
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)

    cfg = OmegaConf.create(
        {
            "geometry": OmegaConf.create(
                {
                    "metrics": {"effective_rank": True, "jer": True},
                    "sample_size": None,
                    "sample_fraction": 0.1,
                }
            ),
            "cache": {"embeddings_dtype": "float16"},
        }
    )
    rows = pipeline.geometry_rows_for_layer(
        embedding_key=key,
        embedding=cache.load_embedding(key),
        labels=None,
        cfg=cfg,
        cache=cache,
        model_id=key.model_id,
        dataset_id=key.dataset_id,
    )

    assert {row["metric_name"] for row in rows} == {"effective_rank"}
    assert "JER requires a model" not in caplog.text


def test_geometry_cache_key_round_trips(tmp_path: Path):
    cache = CacheStore(tmp_path)
    key = _embedding_key(layer=3)
    geom_key = GeometryKey(
        embedding_key=key, metric_name="effective_rank", metric_params_hash=stable_hash("p")
    )
    payload = {"effective_rank": 7.5}
    cache.save_geometry(geom_key, payload)
    assert cache.has_geometry(geom_key)
    assert cache.load_geometry(geom_key) == payload


def test_geometry_rows_for_layer_computes_adjacent_linear_cka(tmp_path: Path):
    cache = CacheStore(tmp_path)
    key1 = _embedding_key(layer=1)
    key2 = _embedding_key(layer=2)
    x = torch.randn(50, 8)
    cache.save_embedding(key1, x)
    cache.save_embedding(key2, 2.0 * x)

    pipeline_path = (
        Path(__file__).resolve().parent.parent / "pipeline" / "02_compute_geometry.py"
    )
    spec = importlib.util.spec_from_file_location("compute_geometry_pipeline4", pipeline_path)
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)

    cfg = OmegaConf.create(
        {
            "geometry": OmegaConf.create(
                {"metrics": {"linear_cka": True}, "sample_size": None, "sample_fraction": 1.0}
            ),
            "cache": {"embeddings_dtype": "float16"},
        }
    )
    first = pipeline.geometry_rows_for_layer(
        embedding_key=key1,
        embedding=cache.load_embedding(key1),
        labels=None,
        cfg=cfg,
        cache=cache,
        model_id=key1.model_id,
        dataset_id=key1.dataset_id,
    )
    assert len(first) == 1
    assert math.isnan(first[0]["metric_value"])
    assert first[0]["failure_reason"] == "missing_reference"

    second = pipeline.geometry_rows_for_layer(
        embedding_key=key2,
        embedding=cache.load_embedding(key2),
        labels=None,
        cfg=cfg,
        cache=cache,
        model_id=key2.model_id,
        dataset_id=key2.dataset_id,
        reference_embedding=cache.load_embedding(key1),
        reference_layer_index=1,
    )
    assert len(second) == 1
    assert second[0]["metric_name"] == "linear_cka"
    assert second[0]["reference_layer_index"] == 1
    assert abs(second[0]["metric_value"] - 1.0) < 1e-5
