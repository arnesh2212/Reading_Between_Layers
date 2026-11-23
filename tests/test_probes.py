"""Tests for the linear and nonlinear probes plus the shared trainer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import Tensor

from reading_between_layers.caching.keys import EmbeddingKey
from reading_between_layers.caching.store import CacheStore
from reading_between_layers.datasets.base import DatasetSample
from reading_between_layers.probing.layer_weighting import (
    LEARNABLE_WEIGHTING_AGGREGATION,
    LEARNABLE_WEIGHTING_LAYER_INDEX,
)
from reading_between_layers.probing.linear import LinearProbe
from reading_between_layers.probing.nonlinear import NonlinearProbe
from reading_between_layers.probing.trainer import train_probe
from reading_between_layers.tasks.classification import ClassificationTask
from reading_between_layers.tasks.clustering import ClusteringTask
from reading_between_layers.tasks.compatibility import TaskType
from reading_between_layers.tasks.multiple_choice_qa import MultipleChoiceQATask


def _probe_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "type": "linear",
            "hidden_size_ratio": 0.5,
            "optimizer": {"name": "adamw", "lr": 5e-3, "weight_decay": 0.0},
            "scheduler": {"name": "linear_warmup", "warmup_ratio": 0.1},
            "training": {
                "max_epochs": 30,
                "early_stopping_patience": 10,
                "batch_size": 32,
                "eval_batch_size": 128,
                "gradient_clip": 1.0,
            },
            "loss": {"name": "cross_entropy", "label_smoothing": 0.0},
            "seeds": [42],
        }
    )


class _AccuracyTask:
    def __init__(self) -> None:
        self.task_type = TaskType.CLASSIFICATION

    def evaluate(self, predictions: Tensor, labels: Tensor) -> dict[str, float]:
        return {"accuracy": float((predictions == labels).float().mean().item())}


def test_linear_probe_param_count_matches_formula():
    cfg = _probe_cfg()
    probe = LinearProbe(
        input_dim=128,
        output_dim=4,
        task_type=TaskType.CLASSIFICATION,
        config=cfg,
    )
    expected = 128 * 4 + 4
    assert probe.num_params == expected


def test_nonlinear_probe_param_count_matches_formula():
    cfg = _probe_cfg()
    probe = NonlinearProbe(
        input_dim=128,
        output_dim=4,
        task_type=TaskType.CLASSIFICATION,
        config=cfg,
    )
    expected = 128 * 64 + 64 + 64 * 4 + 4
    assert probe.num_params == expected


def _build_separable_dataset(
    num_classes: int = 4, per_class: int = 50, dim: int = 32, seed: int = 0
):
    torch.manual_seed(seed)
    centers = torch.randn(num_classes, dim) * 3.0
    xs, ys = [], []
    for c in range(num_classes):
        xs.append(centers[c].unsqueeze(0) + 0.3 * torch.randn(per_class, dim))
        ys.append(torch.full((per_class,), c, dtype=torch.long))
    x = torch.cat(xs)
    y = torch.cat(ys)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed + 1))
    x, y = x[perm], y[perm]
    n = len(x)
    return (
        x[: int(n * 0.6)],
        y[: int(n * 0.6)],
        x[int(n * 0.6) : int(n * 0.8)],
        y[int(n * 0.6) : int(n * 0.8)],
        x[int(n * 0.8) :],
        y[int(n * 0.8) :],
    )


def test_trainer_overfits_separable_classification():
    cfg = _probe_cfg()
    train_x, train_y, val_x, val_y, test_x, test_y = _build_separable_dataset()
    probe = LinearProbe(
        input_dim=train_x.size(1),
        output_dim=4,
        task_type=TaskType.CLASSIFICATION,
        config=cfg,
    )
    result = train_probe(
        embeddings=train_x,
        labels=train_y,
        val_embeddings=val_x,
        val_labels=val_y,
        test_embeddings=test_x,
        test_labels=test_y,
        probe=probe,
        task=_AccuracyTask(),
        config=cfg,
        seed=0,
    )
    assert result["test_metrics"]["accuracy"] > 0.95


def test_trainer_handles_prompt_level_multiple_choice():
    cfg = _probe_cfg()
    train_x, train_y, val_x, val_y, test_x, test_y = _build_separable_dataset(
        num_classes=4,
        per_class=30,
        dim=16,
    )
    probe = LinearProbe(
        input_dim=train_x.size(1),
        output_dim=4,
        task_type=TaskType.MULTIPLE_CHOICE_QA,
        config=cfg,
    )
    result = train_probe(
        embeddings=train_x,
        labels=train_y,
        val_embeddings=val_x,
        val_labels=val_y,
        test_embeddings=test_x,
        test_labels=test_y,
        probe=probe,
        task=MultipleChoiceQATask(num_options=4),
        config=cfg,
        seed=0,
    )
    assert result["test_metrics"]["accuracy"] > 0.90


def test_probe_pipeline_trains_prompt_level_multiple_choice():
    pipeline_path = Path(__file__).resolve().parent.parent / "pipeline" / "03_train_probes.py"
    spec = importlib.util.spec_from_file_location("train_probes_pipeline", pipeline_path)
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)

    cfg = _probe_cfg()
    embeddings = []
    samples = []
    torch.manual_seed(0)
    centers = torch.randn(4, 16) * 3.0
    for label in range(4):
        chunk = centers[label].unsqueeze(0) + 0.2 * torch.randn(30, 16)
        embeddings.append(chunk)
        samples.extend(
            DatasetSample(inputs=[f"question {label}-{i}"], label=label) for i in range(30)
        )
    row = pipeline.train_one_layer(
        layer_index=1,
        embedding=torch.cat(embeddings),
        samples=samples,
        task=MultipleChoiceQATask(num_options=4),
        probe_cfg=cfg,
        seed=0,
    )
    assert row["primary_metric"] > 0.90


def test_learnable_layer_weighting_beats_last_layer_on_synthetic_signal():
    pipeline_path = Path(__file__).resolve().parent.parent / "pipeline" / "03_train_probes.py"
    spec = importlib.util.spec_from_file_location("train_probes_pipeline_learned", pipeline_path)
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)

    cfg = _probe_cfg()
    cfg.training.max_epochs = 8
    cfg.training.early_stopping_patience = 4
    cfg.optimizer.lr = 1e-2
    torch.manual_seed(123)
    n, dim = 120, 8
    labels = torch.arange(n) % 2
    samples = [DatasetSample(inputs=[f"sample {i}"], label=int(labels[i])) for i in range(n)]
    signal = torch.randn(n, dim) * 0.05
    signal[:, 0] = labels.float() * 6.0 - 3.0
    last_noise = torch.randn(n, dim)
    task = ClassificationTask(num_classes=2)

    last = pipeline.train_one_layer(
        layer_index=2,
        embedding=last_noise,
        samples=samples,
        task=task,
        probe_cfg=cfg,
        seed=0,
    )
    learned = pipeline.train_learnable_layer_weighting(
        embeddings_by_layer={1: signal, 2: last_noise},
        samples=samples,
        task=task,
        probe_cfg=cfg,
        seed=0,
    )

    assert learned["layer_index"] == LEARNABLE_WEIGHTING_LAYER_INDEX
    assert learned["layer_aggregation"] == LEARNABLE_WEIGHTING_AGGREGATION
    assert learned["primary_metric"] > last["primary_metric"]
    weights = json.loads(str(learned["learned_layer_weights_json"]))
    assert set(weights) == {"1", "2"}


def _build_noisy_dataset(seed: int = 0):
    """Less separable than the overfitting fixture: centers close, more noise."""
    torch.manual_seed(seed)
    num_classes = 4
    per_class = 60
    dim = 16
    centers = torch.randn(num_classes, dim) * 0.5
    xs, ys = [], []
    for c in range(num_classes):
        xs.append(centers[c].unsqueeze(0) + 1.5 * torch.randn(per_class, dim))
        ys.append(torch.full((per_class,), c, dtype=torch.long))
    x = torch.cat(xs)
    y = torch.cat(ys)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed + 1))
    x, y = x[perm], y[perm]
    n = len(x)
    return (
        x[: int(n * 0.6)],
        y[: int(n * 0.6)],
        x[int(n * 0.6) : int(n * 0.8)],
        y[int(n * 0.6) : int(n * 0.8)],
        x[int(n * 0.8) :],
        y[int(n * 0.8) :],
    )


def test_multi_seed_runs_produce_different_results():
    cfg = _probe_cfg()
    train_x, train_y, val_x, val_y, test_x, test_y = _build_noisy_dataset(seed=1)
    accuracies = []
    for seed in (11, 23, 89):
        probe = LinearProbe(
            input_dim=train_x.size(1),
            output_dim=4,
            task_type=TaskType.CLASSIFICATION,
            config=cfg,
        )
        result = train_probe(
            embeddings=train_x,
            labels=train_y,
            val_embeddings=val_x,
            val_labels=val_y,
            test_embeddings=test_x,
            test_labels=test_y,
            probe=probe,
            task=_AccuracyTask(),
            config=cfg,
            seed=seed,
        )
        accuracies.append(result["test_metrics"]["accuracy"])
    assert len(set(accuracies)) > 1, accuracies


def test_probe_pipeline_handles_clustering_without_supervised_loss(tmp_path: Path, monkeypatch):
    pipeline_path = Path(__file__).resolve().parent.parent / "pipeline" / "03_train_probes.py"
    spec = importlib.util.spec_from_file_location("train_probes_pipeline", pipeline_path)
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)

    cache = CacheStore(tmp_path)
    samples = [DatasetSample(inputs=[f"sample {i}"], label=i % 3) for i in range(30)]
    torch.manual_seed(0)
    embeddings = torch.randn(30, 12) + torch.tensor(
        [[i % 3] * 12 for i in range(30)], dtype=torch.float32
    )
    keys = {
        layer: EmbeddingKey(
            model_id="test/model",
            dataset_id="test/dataset",
            dataset_subset=None,
            split="train",
            layer_index=layer,
            pooling_name="mean",
            dtype="float16",
            extraction_max_length=32,
            truncation=True,
        )
        for layer in (1, 2)
    }
    for key in keys.values():
        cache.save_embedding(key, embeddings.half())

    cfg = OmegaConf.create({"seed": 0, "probe": _probe_cfg()})
    cfg.probe.seeds = [7, 11]
    monkeypatch.setenv("CLUSTERING_EVAL_LIMIT", "20")

    rows = pipeline.run_probes(
        cfg=cfg,
        cache=cache,
        embedding_keys=keys,
        samples=samples,
        task=ClusteringTask(num_clusters=3),
    )

    assert len(rows) == 4
    assert {row["seed"] for row in rows} == {7, 11}
    assert all(row["num_params"] == 0 for row in rows)
    assert all(0.0 <= row["primary_metric"] <= 1.0 for row in rows)


def test_run_probes_writes_learnable_row_only_for_default_pooling(tmp_path: Path):
    pipeline_path = Path(__file__).resolve().parent.parent / "pipeline" / "03_train_probes.py"
    spec = importlib.util.spec_from_file_location(
        "train_probes_pipeline_default_only", pipeline_path
    )
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "pooling": {"name": "cls"},
            "task": {"pooling_default": "cls"},
            "probe": _probe_cfg(),
        }
    )
    cfg.probe.seeds = [5]
    cfg.probe.training.max_epochs = 2
    cfg.probe.training.early_stopping_patience = 1
    cache = CacheStore(tmp_path)
    samples = [DatasetSample(inputs=[f"sample {i}"], label=i % 2) for i in range(40)]
    keys = {
        layer: EmbeddingKey(
            model_id="test/model",
            dataset_id="test/dataset",
            dataset_subset=None,
            split="train",
            layer_index=layer,
            pooling_name="cls",
            dtype="float16",
            extraction_max_length=32,
            truncation=True,
        )
        for layer in (1, 2)
    }
    for layer, key in keys.items():
        torch.manual_seed(layer)
        cache.save_embedding(key, torch.randn(40, 6).half())

    rows = pipeline.run_probes(
        cfg=cfg,
        cache=cache,
        embedding_keys=keys,
        samples=samples,
        task=ClassificationTask(num_classes=2),
    )
    learned_rows = [
        r
        for r in rows
        if r.get("layer_aggregation") == LEARNABLE_WEIGHTING_AGGREGATION
    ]
    assert len(learned_rows) == 1

    cfg.pooling.name = "mean"
    rows = pipeline.run_probes(
        cfg=cfg,
        cache=cache,
        embedding_keys=keys,
        samples=samples,
        task=ClassificationTask(num_classes=2),
    )
    learned_rows = [
        r
        for r in rows
        if r.get("layer_aggregation") == LEARNABLE_WEIGHTING_AGGREGATION
    ]
    assert learned_rows == []


def test_run_probes_supervised_uses_mmap_batch_loader(tmp_path: Path, monkeypatch):
    pipeline_path = Path(__file__).resolve().parent.parent / "pipeline" / "03_train_probes.py"
    spec = importlib.util.spec_from_file_location("train_probes_pipeline_mmap", pipeline_path)
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "pooling": {"name": "cls"},
            "task": {"pooling_default": "cls"},
            "probe": _probe_cfg(),
        }
    )
    cfg.probe.seeds = [3]
    cfg.probe.training.max_epochs = 2
    cfg.probe.training.early_stopping_patience = 1
    cfg.probe.training.batch_size = 8
    cfg.probe.training.eval_batch_size = 8
    cache = CacheStore(tmp_path)
    labels = torch.arange(32) % 2
    samples = [DatasetSample(inputs=[f"sample {i}"], label=int(labels[i])) for i in range(32)]
    keys = {
        layer: EmbeddingKey(
            model_id="test/model",
            dataset_id="test/dataset",
            dataset_subset=None,
            split="train",
            layer_index=layer,
            pooling_name="cls",
            dtype="float16",
            extraction_max_length=32,
            truncation=True,
        )
        for layer in (1, 2)
    }
    signal = torch.randn(32, 6) * 0.05
    signal[:, 0] = labels.float() * 4.0 - 2.0
    cache.save_embedding(keys[1], signal.half())
    cache.save_embedding(keys[2], torch.randn(32, 6).half())

    def fail_full_load(_key):
        raise AssertionError("supervised probe training should use mmap batch loading")

    monkeypatch.setattr(cache, "load_embedding", fail_full_load)

    rows = pipeline.run_probes(
        cfg=cfg,
        cache=cache,
        embedding_keys=keys,
        samples=samples,
        task=ClassificationTask(num_classes=2),
    )

    assert len(rows) == 3
    assert any(row.get("layer_aggregation") == LEARNABLE_WEIGHTING_AGGREGATION for row in rows)
