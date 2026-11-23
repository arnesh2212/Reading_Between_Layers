"""Tests for the sweep resolver and manifest IO."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from reading_between_layers.caching.keys import EmbeddingKey
from reading_between_layers.caching.store import CacheStore
from reading_between_layers.datasets.registry import get_dataset_spec
from reading_between_layers.models.registry import MODEL_REGISTRY, get_model_spec
from reading_between_layers.tasks.compatibility import TaskType, check_compatibility
from reading_between_layers.utils.manifest import Manifest, read_manifest, write_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = str(REPO_ROOT / "configs")
RESOLVER_PATH = REPO_ROOT / "pipeline" / "00_resolve_sweep.py"


def _load_resolver():
    spec = importlib.util.spec_from_file_location("resolver_module", RESOLVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


resolver = _load_resolver()


@pytest.fixture(autouse=True)
def _clear_hydra():
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()


def _compose_paper_sweep(cache_root: Path):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(
            config_name="resolver_config",
            overrides=[
                "experiment=paper",
                f"cache.root={cache_root}",
            ],
        )


def test_paper_sweep_produces_non_empty_tuple_list(tmp_path: Path):
    cfg = _compose_paper_sweep(tmp_path)
    manifest = resolver.resolve_sweep(cfg, cache_root=tmp_path)
    assert manifest.tuples


def test_every_resolved_tuple_passes_compatibility(tmp_path: Path):
    cfg = _compose_paper_sweep(tmp_path)
    manifest = resolver.resolve_sweep(cfg, cache_root=tmp_path)
    for t in manifest.tuples:
        model_spec = get_model_spec(t["model"])
        check_compatibility(model_spec, TaskType(t["task"]))


def test_decoder_retrieval_filtered_out(tmp_path: Path):
    cfg = _compose_paper_sweep(tmp_path)
    manifest = resolver.resolve_sweep(cfg, cache_root=tmp_path)
    for t in manifest.tuples:
        if t["task"] == TaskType.RETRIEVAL.value:
            assert get_model_spec(t["model"]).family != "decoder", t


def test_cls_is_rewritten_for_encoder_decoder_and_ssm_tuples(tmp_path: Path):
    cfg = _compose_paper_sweep(tmp_path)
    manifest = resolver.resolve_sweep(cfg, cache_root=tmp_path)
    for t in manifest.tuples:
        family = get_model_spec(t["model"]).family
        if family in {"encoder_decoder", "ssm"}:
            assert t["pooling"] != "cls", t


def test_datasets_by_family_limits_resolved_tuples(tmp_path: Path):
    cfg = _compose_paper_sweep(tmp_path)
    manifest = resolver.resolve_sweep(cfg, cache_root=tmp_path)
    policy = OmegaConf.to_object(cfg.experiment.datasets_by_family)

    for t in manifest.tuples:
        model_spec = get_model_spec(t["model"])
        allowed = policy.get(model_spec.family)
        if allowed is not None:
            assert t["dataset"] in allowed, t

    gpt2_datasets = {t["dataset"] for t in manifest.tuples if t["model"] == "gpt2"}
    assert gpt2_datasets == {"arc", "hellaswag", "piqa"}


def test_vram_estimate_non_zero_for_every_model():
    for spec in MODEL_REGISTRY.values():
        assert resolver.estimate_vram_bytes(spec) > 0


def test_manifest_writes_valid_json_and_reads_back_equal(tmp_path: Path):
    m = Manifest(
        run_id="run-1",
        timestamp=datetime.now(UTC).isoformat(),
        git_commit="abc1234",
        experiment="paper",
        tuples=[
            {
                "model": "bert_base_uncased",
                "dataset": "sst2",
                "task": "classification",
                "probe": "linear",
                "pooling": "cls",
                "seed": 42,
                "vram_estimate_bytes": 1234,
            }
        ],
        estimates={"total_tuples": 1, "estimated_cache_delta_bytes": 100},
        warnings=["one warning"],
        commands=["python pipeline/01_extract_embeddings.py --multirun"],
    )
    path = write_manifest(m, tmp_path / "run")
    loaded = read_manifest(path)
    assert loaded == m
    assert loaded.hash() == m.hash()


def test_cache_hit_tuples_appear_in_warnings(tmp_path: Path):
    cfg = _compose_paper_sweep(tmp_path)
    model_spec = get_model_spec("bert_base_uncased")
    dataset_spec = get_dataset_spec("sst2")
    cache = CacheStore(tmp_path)
    key = EmbeddingKey(
        model_id=model_spec.hf_id,
        dataset_id=dataset_spec.hf_id,
        dataset_subset=dataset_spec.hf_subset,
        split=dataset_spec.splits_available[0],
        layer_index=resolver._first_extracted_layer(cfg),
        pooling_name="cls",
        dtype=str(cfg.cache.embeddings_dtype),
        extraction_max_length=int(cfg.extraction.max_length),
        truncation=bool(cfg.extraction.truncation),
        sample_order=resolver._cache_sample_order(cfg),
    )
    cache.save_embedding(key, torch.zeros(1, dtype=torch.float16))

    manifest = resolver.resolve_sweep(cfg, cache_root=tmp_path)
    assert any("already cached" in w for w in manifest.warnings)
    for t in manifest.tuples:
        if (
            t["model"] == "bert_base_uncased"
            and t["dataset"] == "sst2"
            and t["pooling"] == "cls"
            and t["task"] == "classification"
        ):
            pytest.fail(f"cached tuple should have been skipped: {t}")
