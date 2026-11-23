"""Hydra entry point: train and evaluate probes per layer and per seed."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reading_between_layers.exceptions import InvalidConfigError  # noqa: E402
from reading_between_layers.utils.logging_config import configure_logging  # noqa: E402
from reading_between_layers.utils.manifest import find_latest_manifest_hash  # noqa: E402

configure_logging()

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from reading_between_layers.caching.keys import EmbeddingKey
    from reading_between_layers.caching.store import CacheStore
    from reading_between_layers.datasets.base import DatasetSample
    from reading_between_layers.tasks.base import Task


def _check_manifest_gate() -> None:
    """Refuse to multirun without a fresh manifest from pipeline/00_resolve_sweep.py."""
    if "--multirun" not in sys.argv:
        return
    expected = os.environ.get("MANIFEST_HASH")
    latest = find_latest_manifest_hash(REPO_ROOT / "outputs" / "runs")
    if not expected or expected != latest:
        raise InvalidConfigError("Run pipeline/00_resolve_sweep.py first")


_check_manifest_gate()

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from reading_between_layers.caching.keys import EmbeddingKey, ProbeKey  # noqa: E402
from reading_between_layers.caching.store import CacheStore  # noqa: E402
from reading_between_layers.datasets.base import DatasetSample  # noqa: E402
from reading_between_layers.datasets.materialize import iter_hf_samples  # noqa: E402
from reading_between_layers.datasets.registry import get_dataset_spec  # noqa: E402
from reading_between_layers.datasets.splits import resolve_train_split  # noqa: E402
from reading_between_layers.models.registry import get_model_spec  # noqa: E402
from reading_between_layers.pooling.selection import (  # noqa: E402
    normalize_pooling_name,
    representation_side_for_model,
)
from reading_between_layers.probing.heads import make_head  # noqa: E402
from reading_between_layers.probing.layer_weighting import (  # noqa: E402
    LEARNABLE_WEIGHTING_AGGREGATION,
    LEARNABLE_WEIGHTING_LAYER_INDEX,
    SINGLE_LAYER_AGGREGATION,
    TRAINABLE_LAYER_WEIGHTING_PROBES,
    TRAINABLE_LAYER_WEIGHTING_TASKS,
    LayerWeightedProbe,
)
from reading_between_layers.probing.registry import PROBE_REGISTRY  # noqa: E402
from reading_between_layers.probing.trainer import train_probe, train_probe_from_loaders  # noqa: E402
from reading_between_layers.tasks.classification import ClassificationTask  # noqa: E402
from reading_between_layers.tasks.clustering import ClusteringTask  # noqa: E402
from reading_between_layers.tasks.compatibility import TaskType, check_compatibility  # noqa: E402
from reading_between_layers.tasks.generation import GenerationTask  # noqa: E402
from reading_between_layers.tasks.multiple_choice_qa import MultipleChoiceQATask  # noqa: E402
from reading_between_layers.tasks.pair_classification import PairClassificationTask  # noqa: E402
from reading_between_layers.tasks.qa import ExtractiveQATask, GenerativeQATask  # noqa: E402
from reading_between_layers.tasks.retrieval import RetrievalTask  # noqa: E402
from reading_between_layers.tasks.sts import STSTask  # noqa: E402
from reading_between_layers.tasks.token_classification import TokenClassificationTask  # noqa: E402
from reading_between_layers.utils.hashing import stable_hash  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CLUSTERING_EVAL_LIMIT = 10_000


def build_task(task_type: TaskType, num_classes: int | None) -> Task:
    """Construct the right Task instance for the configured task type."""
    if task_type == TaskType.CLASSIFICATION:
        return ClassificationTask(num_classes=num_classes or 2)
    if task_type == TaskType.PAIR_CLASSIFICATION:
        return PairClassificationTask(num_classes=num_classes or 2)
    if task_type == TaskType.STS:
        return STSTask()
    if task_type == TaskType.TOKEN_CLASSIFICATION:
        return TokenClassificationTask(label_names=[f"L{i}" for i in range(num_classes or 2)])
    if task_type == TaskType.RETRIEVAL:
        return RetrievalTask()
    if task_type == TaskType.CLUSTERING:
        return ClusteringTask(num_clusters=num_classes or 2)
    if task_type == TaskType.MULTIPLE_CHOICE_QA:
        return MultipleChoiceQATask(num_options=num_classes or 4)
    if task_type == TaskType.QA_EXTRACTIVE:
        return ExtractiveQATask()
    if task_type == TaskType.QA_GENERATIVE:
        return GenerativeQATask()
    if task_type == TaskType.GENERATION:
        return GenerationTask()
    raise ValueError(f"unknown task type {task_type!r}")


def split_indices(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministically partition ``range(n)`` into 60/20/20 train/val/test."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def _probe_labels(task: Task, samples: list[DatasetSample]) -> torch.Tensor:
    task_type = task.spec.task_type
    if task_type == TaskType.STS:
        return torch.tensor([float(s.label) for s in samples], dtype=torch.float32)
    if task_type in {
        TaskType.CLASSIFICATION,
        TaskType.PAIR_CLASSIFICATION,
        TaskType.MULTIPLE_CHOICE_QA,
    }:
        labels = torch.tensor([int(s.label) for s in samples], dtype=torch.long)
        num_classes = getattr(task, "num_classes", None) or getattr(task, "num_options", None)
        if labels.numel() and num_classes is not None:
            min_label = int(labels.min())
            max_label = int(labels.max())
            if min_label < 0 or max_label >= int(num_classes):
                raise InvalidConfigError(
                    f"{task_type.value} labels must be in [0, {int(num_classes)}); "
                    f"got min={min_label}, max={max_label}"
                )
        return labels
    raise InvalidConfigError(
        f"batched cached-embedding probe loading is not implemented for {task_type.value}"
    )


def _lazy_probe_supported(task: Task) -> bool:
    return task.spec.task_type in {
        TaskType.CLASSIFICATION,
        TaskType.PAIR_CLASSIFICATION,
        TaskType.MULTIPLE_CHOICE_QA,
        TaskType.STS,
    }


def _embedding_num_options(
    embedding: torch.Tensor,
    samples: list[DatasetSample],
    task: Task,
) -> int | None:
    rows = int(embedding.shape[0])
    sample_rows = len(samples)
    if rows == sample_rows:
        return None
    if task.spec.task_type == TaskType.MULTIPLE_CHOICE_QA:
        num_options = int(getattr(task, "num_options", 0))
        if num_options > 0 and rows == sample_rows * num_options:
            return num_options
    raise InvalidConfigError(
        "cached embedding rows do not align with probe samples: "
        f"embedding rows={rows}, samples={sample_rows}, task={task.spec.task_type.value}"
    )


def _feature_shape_for_loader(
    embedding: torch.Tensor,
    samples: list[DatasetSample],
    task: Task,
    *,
    num_layers: int = 1,
) -> tuple[int, ...]:
    hidden = int(embedding.shape[-1])
    num_options = _embedding_num_options(embedding, samples, task)
    if num_layers == 1:
        if num_options is None:
            return (hidden,)
        return (num_options, hidden)
    if num_options is None:
        return (num_layers, hidden)
    return (num_options, num_layers, hidden)


class _CachedEmbeddingBatchLoader:
    """Yield probe minibatches from mmap-backed cached embeddings."""

    def __init__(
        self,
        embeddings: torch.Tensor | Mapping[int, torch.Tensor],
        labels: torch.Tensor,
        indices: torch.Tensor,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        num_options: int | None = None,
    ) -> None:
        if int(batch_size) <= 0:
            raise InvalidConfigError(f"probe batch_size must be positive, got {batch_size}")
        self.embeddings = embeddings
        self.labels = labels.detach().cpu()
        self.indices = indices.detach().cpu().long()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_options = num_options
        if isinstance(embeddings, Mapping):
            self.layer_indices = [int(layer) for layer in sorted(embeddings)]
            if not self.layer_indices:
                raise InvalidConfigError("learnable weighting needs at least one layer")
        else:
            self.layer_indices = None

    def __len__(self) -> int:
        if self.indices.numel() == 0:
            return 0
        return (int(self.indices.numel()) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = self.indices
        if self.shuffle and order.numel() > 1:
            perm = torch.randperm(order.numel(), generator=torch.Generator().manual_seed(self.seed))
            order = order[perm]
        for start in range(0, int(order.numel()), self.batch_size):
            batch_indices = order[start : start + self.batch_size]
            yield self._read_features(batch_indices), self.labels[batch_indices]

    def _read_tensor_rows(
        self,
        embedding: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_options is None:
            return embedding[sample_indices]
        option_offsets = torch.arange(self.num_options, dtype=torch.long)
        rows = (sample_indices[:, None] * self.num_options + option_offsets[None, :]).reshape(-1)
        return embedding[rows].reshape(
            sample_indices.numel(),
            self.num_options,
            embedding.shape[-1],
        )

    def _read_features(self, sample_indices: torch.Tensor) -> torch.Tensor:
        if isinstance(self.embeddings, Mapping):
            assert self.layer_indices is not None
            per_layer = [
                self._read_tensor_rows(self.embeddings[layer], sample_indices)
                for layer in self.layer_indices
            ]
            return torch.stack(per_layer, dim=-2)
        return self._read_tensor_rows(self.embeddings, sample_indices)


def _make_loader(
    embeddings: torch.Tensor | Mapping[int, torch.Tensor],
    labels: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_options: int | None,
) -> _CachedEmbeddingBatchLoader:
    return _CachedEmbeddingBatchLoader(
        embeddings=embeddings,
        labels=labels,
        indices=indices,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        num_options=num_options,
    )


def _make_probe(task: Task, input_dim: int, probe_cfg: DictConfig, feature_rank: int = 2):
    probe_cls = PROBE_REGISTRY[probe_cfg.type]
    num_outputs = getattr(task, "num_classes", None) or getattr(task, "num_options", None)
    output_dim = (
        1
        if task.spec.task_type == TaskType.MULTIPLE_CHOICE_QA and feature_rank == 3
        else num_outputs or 1
    )
    probe = probe_cls(
        input_dim=input_dim,
        output_dim=output_dim,
        task_type=task.spec.task_type,
        config=probe_cfg,
    )
    # Validate the head shape matches what make_head would produce; we use the
    # probe class directly because PROBE_REGISTRY already returns one with the
    # right output head fused in.
    _ = make_head(task.spec.task_type, input_dim, output_dim, probe_cfg)
    return probe


def _make_layer_weighted_probe(
    task: Task,
    input_dim: int,
    num_layers: int,
    probe_cfg: DictConfig,
    mixed_feature_rank: int,
) -> LayerWeightedProbe:
    base_probe = _make_probe(task, input_dim, probe_cfg, feature_rank=mixed_feature_rank)
    return LayerWeightedProbe(base_probe=base_probe, num_layers=num_layers)


def _probe_result_row(
    layer_index: int,
    seed: int,
    result: dict,
    task: Task,
    *,
    layer_aggregation: str = SINGLE_LAYER_AGGREGATION,
) -> dict[str, float | int | str]:
    return {
        "layer_index": layer_index,
        "seed": seed,
        "primary_metric": result["test_metrics"][task.spec.metric_primary],
        "secondary_metrics_json": str(
            {k: v for k, v in result["test_metrics"].items() if k != task.spec.metric_primary}
        ),
        "ci_lower": result["test_primary_ci_lo"],
        "ci_upper": result["test_primary_ci_hi"],
        "num_params": int(result["num_params"]),
        "val_primary": result["val_primary"],
        "layer_aggregation": layer_aggregation,
    }


def train_one_layer(
    layer_index: int,
    embedding: torch.Tensor,
    samples: list[DatasetSample],
    task: Task,
    probe_cfg: DictConfig,
    seed: int,
    device: str | torch.device | None = None,
) -> dict[str, float | int | str]:
    """Train one probe using a deterministic train/val/test split of one split."""
    x, y = task.build_features(samples, embedding.float())
    train_idx, val_idx, test_idx = split_indices(x.size(0), seed=seed)
    probe = _make_probe(task, x.size(-1), probe_cfg, x.dim())
    result = train_probe(
        embeddings=x[train_idx],
        labels=y[train_idx],
        val_embeddings=x[val_idx],
        val_labels=y[val_idx],
        test_embeddings=x[test_idx],
        test_labels=y[test_idx],
        probe=probe,
        task=task,
        config=probe_cfg,
        seed=seed,
        device=device,
    )
    row = _probe_result_row(layer_index, seed, result, task)
    return row


def train_one_layer_official_test(
    layer_index: int,
    train_embedding: torch.Tensor,
    train_samples: list[DatasetSample],
    test_embedding: torch.Tensor,
    test_samples: list[DatasetSample],
    task: Task,
    probe_cfg: DictConfig,
    seed: int,
    device: str | torch.device | None = None,
) -> dict[str, float | int | str]:
    """Train on the configured train split and select/evaluate on test."""
    x_train, y_train = task.build_features(train_samples, train_embedding.float())
    x_test, y_test = task.build_features(test_samples, test_embedding.float())
    if x_train.size(-1) != x_test.size(-1):
        raise InvalidConfigError(
            f"train/test feature dims differ for layer {layer_index}: "
            f"{x_train.size(-1)} vs {x_test.size(-1)}"
        )
    probe = _make_probe(task, x_train.size(-1), probe_cfg, x_train.dim())
    result = train_probe(
        embeddings=x_train,
        labels=y_train,
        val_embeddings=x_test,
        val_labels=y_test,
        test_embeddings=x_test,
        test_labels=y_test,
        probe=probe,
        task=task,
        config=probe_cfg,
        seed=seed,
        device=device,
    )
    row = _probe_result_row(layer_index, seed, result, task)
    row["split_strategy"] = "official_test"
    return row


def train_one_layer_batched(
    layer_index: int,
    embedding: torch.Tensor,
    samples: list[DatasetSample],
    task: Task,
    probe_cfg: DictConfig,
    seed: int,
    device: str | torch.device | None = None,
) -> dict[str, float | int | str]:
    """Train one probe from mmap-backed cached embeddings in minibatches."""
    labels = _probe_labels(task, samples)
    num_options = _embedding_num_options(embedding, samples, task)
    feature_shape = _feature_shape_for_loader(embedding, samples, task)
    train_idx, val_idx, test_idx = split_indices(labels.size(0), seed=seed)
    batch_size = int(probe_cfg.training.batch_size)
    eval_batch_size = int(probe_cfg.training.get("eval_batch_size", batch_size))
    train_loader = _make_loader(
        embedding,
        labels,
        train_idx,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_options=num_options,
    )
    val_loader = _make_loader(
        embedding,
        labels,
        val_idx,
        batch_size=eval_batch_size,
        shuffle=False,
        seed=seed,
        num_options=num_options,
    )
    test_loader = _make_loader(
        embedding,
        labels,
        test_idx,
        batch_size=eval_batch_size,
        shuffle=False,
        seed=seed,
        num_options=num_options,
    )
    probe = _make_probe(
        task,
        input_dim=feature_shape[-1],
        probe_cfg=probe_cfg,
        feature_rank=len(feature_shape) + 1,
    )
    result = train_probe_from_loaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        probe=probe,
        task=task,
        config=probe_cfg,
        seed=seed,
        device=device,
    )
    return _probe_result_row(layer_index, seed, result, task)


def train_one_layer_official_test_batched(
    layer_index: int,
    train_embedding: torch.Tensor,
    train_samples: list[DatasetSample],
    test_embedding: torch.Tensor,
    test_samples: list[DatasetSample],
    task: Task,
    probe_cfg: DictConfig,
    seed: int,
    device: str | torch.device | None = None,
) -> dict[str, float | int | str]:
    """Train/evaluate official-test probes from mmap-backed cached embeddings."""
    train_labels = _probe_labels(task, train_samples)
    test_labels = _probe_labels(task, test_samples)
    train_num_options = _embedding_num_options(train_embedding, train_samples, task)
    test_num_options = _embedding_num_options(test_embedding, test_samples, task)
    if train_num_options != test_num_options:
        raise InvalidConfigError(
            f"train/test option layouts differ for layer {layer_index}: "
            f"{train_num_options} vs {test_num_options}"
        )
    train_feature_shape = _feature_shape_for_loader(train_embedding, train_samples, task)
    test_feature_shape = _feature_shape_for_loader(test_embedding, test_samples, task)
    if train_feature_shape != test_feature_shape:
        raise InvalidConfigError(
            f"train/test feature shapes differ for layer {layer_index}: "
            f"{train_feature_shape} vs {test_feature_shape}"
        )
    batch_size = int(probe_cfg.training.batch_size)
    eval_batch_size = int(probe_cfg.training.get("eval_batch_size", batch_size))
    train_idx = torch.arange(train_labels.size(0), dtype=torch.long)
    test_idx = torch.arange(test_labels.size(0), dtype=torch.long)
    train_loader = _make_loader(
        train_embedding,
        train_labels,
        train_idx,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_options=train_num_options,
    )
    test_loader = _make_loader(
        test_embedding,
        test_labels,
        test_idx,
        batch_size=eval_batch_size,
        shuffle=False,
        seed=seed,
        num_options=test_num_options,
    )
    probe = _make_probe(
        task,
        input_dim=train_feature_shape[-1],
        probe_cfg=probe_cfg,
        feature_rank=len(train_feature_shape) + 1,
    )
    result = train_probe_from_loaders(
        train_loader=train_loader,
        val_loader=test_loader,
        test_loader=test_loader,
        probe=probe,
        task=task,
        config=probe_cfg,
        seed=seed,
        device=device,
    )
    row = _probe_result_row(layer_index, seed, result, task)
    row["split_strategy"] = "official_test"
    return row


def _stack_layer_features(
    embeddings_by_layer: dict[int, torch.Tensor],
    samples: list[DatasetSample],
    task: Task,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Build task features per layer and stack them for scalar-mix training."""
    layer_indices = sorted(embeddings_by_layer)
    if not layer_indices:
        raise InvalidConfigError("learnable layer weighting needs at least one layer")
    features: list[torch.Tensor] = []
    labels_ref: torch.Tensor | None = None
    feature_shape: torch.Size | None = None
    for layer_index in layer_indices:
        x, y = task.build_features(samples, embeddings_by_layer[layer_index].float())
        if x.ndim < 2:
            raise InvalidConfigError(
                "learnable layer weighting expects feature tensors with a hidden dimension; "
                f"layer {layer_index} produced shape {tuple(x.shape)}"
            )
        if labels_ref is None:
            labels_ref = y
            feature_shape = x.shape
        else:
            if not torch.equal(labels_ref, y):
                raise InvalidConfigError(
                    f"labels changed while building layer {layer_index} features"
                )
            if x.shape != feature_shape:
                expected = tuple(feature_shape) if feature_shape is not None else ()
                raise InvalidConfigError(
                    "feature shapes differ across layers for learnable weighting: "
                    f"expected {expected}, got {tuple(x.shape)}"
                )
        features.append(x)
    assert labels_ref is not None
    stack_dim = features[0].ndim - 1
    return torch.stack(features, dim=stack_dim), labels_ref, layer_indices


def _remap_layer_weights(
    probe: LayerWeightedProbe,
    layer_indices: list[int],
) -> dict[int, float]:
    temporary = probe.layer_weights()
    return {int(layer): float(temporary[i]) for i, layer in enumerate(layer_indices)}


def _annotate_learnable_row(
    row: dict[str, float | int | str],
    layer_indices: list[int],
    layer_weights: dict[int, float],
) -> dict[str, float | int | str]:
    row["source_layer_indices_json"] = json.dumps([int(layer) for layer in layer_indices])
    row["learned_layer_weights_json"] = json.dumps(
        {str(layer): float(layer_weights[layer]) for layer in layer_indices},
        sort_keys=True,
    )
    row["argmax_weight_layer"] = int(max(layer_weights, key=layer_weights.get))
    row["num_layers"] = int(len(layer_indices))
    return row


def train_learnable_layer_weighting(
    embeddings_by_layer: dict[int, torch.Tensor],
    samples: list[DatasetSample],
    task: Task,
    probe_cfg: DictConfig,
    seed: int,
    device: str | torch.device | None = None,
) -> dict[str, float | int | str]:
    """Train one all-layer scalar mix using a deterministic train/val/test split."""
    x, y, layer_indices = _stack_layer_features(embeddings_by_layer, samples, task)
    train_idx, val_idx, test_idx = split_indices(x.size(0), seed=seed)
    probe = _make_layer_weighted_probe(
        task=task,
        input_dim=x.size(-1),
        num_layers=len(layer_indices),
        probe_cfg=probe_cfg,
        mixed_feature_rank=x.ndim - 1,
    )
    result = train_probe(
        embeddings=x[train_idx],
        labels=y[train_idx],
        val_embeddings=x[val_idx],
        val_labels=y[val_idx],
        test_embeddings=x[test_idx],
        test_labels=y[test_idx],
        probe=probe,
        task=task,
        config=probe_cfg,
        seed=seed,
        device=device,
    )
    row = _probe_result_row(
        LEARNABLE_WEIGHTING_LAYER_INDEX,
        seed,
        result,
        task,
        layer_aggregation=LEARNABLE_WEIGHTING_AGGREGATION,
    )
    return _annotate_learnable_row(row, layer_indices, _remap_layer_weights(probe, layer_indices))


def train_learnable_layer_weighting_official_test(
    train_embeddings_by_layer: dict[int, torch.Tensor],
    train_samples: list[DatasetSample],
    test_embeddings_by_layer: dict[int, torch.Tensor],
    test_samples: list[DatasetSample],
    task: Task,
    probe_cfg: DictConfig,
    seed: int,
    device: str | torch.device | None = None,
) -> dict[str, float | int | str]:
    """Train a scalar mix on the train split and evaluate on the official test split."""
    x_train, y_train, layer_indices = _stack_layer_features(
        train_embeddings_by_layer,
        train_samples,
        task,
    )
    x_test, y_test, test_layer_indices = _stack_layer_features(
        test_embeddings_by_layer,
        test_samples,
        task,
    )
    if layer_indices != test_layer_indices:
        raise InvalidConfigError(
            "train/test layer sets differ for learnable weighting: "
            f"{layer_indices} vs {test_layer_indices}"
        )
    if x_train.size(-1) != x_test.size(-1):
        raise InvalidConfigError(
            "train/test feature dims differ for learnable weighting: "
            f"{x_train.size(-1)} vs {x_test.size(-1)}"
        )
    probe = _make_layer_weighted_probe(
        task=task,
        input_dim=x_train.size(-1),
        num_layers=len(layer_indices),
        probe_cfg=probe_cfg,
        mixed_feature_rank=x_train.ndim - 1,
    )
    result = train_probe(
        embeddings=x_train,
        labels=y_train,
        val_embeddings=x_test,
        val_labels=y_test,
        test_embeddings=x_test,
        test_labels=y_test,
        probe=probe,
        task=task,
        config=probe_cfg,
        seed=seed,
        device=device,
    )
    row = _probe_result_row(
        LEARNABLE_WEIGHTING_LAYER_INDEX,
        seed,
        result,
        task,
        layer_aggregation=LEARNABLE_WEIGHTING_AGGREGATION,
    )
    row["split_strategy"] = "official_test"
    return _annotate_learnable_row(row, layer_indices, _remap_layer_weights(probe, layer_indices))


def _validate_layer_embedding_layouts(
    embeddings_by_layer: Mapping[int, torch.Tensor],
    samples: list[DatasetSample],
    task: Task,
) -> tuple[list[int], int | None, tuple[int, ...]]:
    layer_indices = [int(layer) for layer in sorted(embeddings_by_layer)]
    if not layer_indices:
        raise InvalidConfigError("learnable layer weighting needs at least one layer")
    first = embeddings_by_layer[layer_indices[0]]
    num_options = _embedding_num_options(first, samples, task)
    feature_shape = _feature_shape_for_loader(
        first,
        samples,
        task,
        num_layers=len(layer_indices),
    )
    first_rows = int(first.shape[0])
    first_hidden = int(first.shape[-1])
    for layer in layer_indices[1:]:
        embedding = embeddings_by_layer[layer]
        layer_options = _embedding_num_options(embedding, samples, task)
        if layer_options != num_options:
            raise InvalidConfigError(
                "feature layouts differ across layers for learnable weighting: "
                f"layer {layer} has num_options={layer_options}, expected {num_options}"
            )
        if int(embedding.shape[0]) != first_rows or int(embedding.shape[-1]) != first_hidden:
            raise InvalidConfigError(
                "embedding shapes differ across layers for learnable weighting: "
                f"layer {layer} has shape {tuple(embedding.shape)}, "
                f"expected rows={first_rows}, hidden={first_hidden}"
            )
    return layer_indices, num_options, feature_shape


def train_learnable_layer_weighting_batched(
    embeddings_by_layer: Mapping[int, torch.Tensor],
    samples: list[DatasetSample],
    task: Task,
    probe_cfg: DictConfig,
    seed: int,
    device: str | torch.device | None = None,
) -> dict[str, float | int | str]:
    """Train scalar-mix probes from mmap-backed cached layers in minibatches."""
    labels = _probe_labels(task, samples)
    layer_indices, num_options, feature_shape = _validate_layer_embedding_layouts(
        embeddings_by_layer,
        samples,
        task,
    )
    train_idx, val_idx, test_idx = split_indices(labels.size(0), seed=seed)
    batch_size = int(probe_cfg.training.batch_size)
    eval_batch_size = int(probe_cfg.training.get("eval_batch_size", batch_size))
    train_loader = _make_loader(
        embeddings_by_layer,
        labels,
        train_idx,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_options=num_options,
    )
    val_loader = _make_loader(
        embeddings_by_layer,
        labels,
        val_idx,
        batch_size=eval_batch_size,
        shuffle=False,
        seed=seed,
        num_options=num_options,
    )
    test_loader = _make_loader(
        embeddings_by_layer,
        labels,
        test_idx,
        batch_size=eval_batch_size,
        shuffle=False,
        seed=seed,
        num_options=num_options,
    )
    probe = _make_layer_weighted_probe(
        task=task,
        input_dim=feature_shape[-1],
        num_layers=len(layer_indices),
        probe_cfg=probe_cfg,
        mixed_feature_rank=len(feature_shape),
    )
    result = train_probe_from_loaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        probe=probe,
        task=task,
        config=probe_cfg,
        seed=seed,
        device=device,
    )
    row = _probe_result_row(
        LEARNABLE_WEIGHTING_LAYER_INDEX,
        seed,
        result,
        task,
        layer_aggregation=LEARNABLE_WEIGHTING_AGGREGATION,
    )
    return _annotate_learnable_row(row, layer_indices, _remap_layer_weights(probe, layer_indices))


def train_learnable_layer_weighting_official_test_batched(
    train_embeddings_by_layer: Mapping[int, torch.Tensor],
    train_samples: list[DatasetSample],
    test_embeddings_by_layer: Mapping[int, torch.Tensor],
    test_samples: list[DatasetSample],
    task: Task,
    probe_cfg: DictConfig,
    seed: int,
    device: str | torch.device | None = None,
) -> dict[str, float | int | str]:
    """Train official-test scalar-mix probes from mmap-backed cached layers."""
    train_labels = _probe_labels(task, train_samples)
    test_labels = _probe_labels(task, test_samples)
    layer_indices, train_num_options, train_feature_shape = _validate_layer_embedding_layouts(
        train_embeddings_by_layer,
        train_samples,
        task,
    )
    test_layer_indices, test_num_options, test_feature_shape = _validate_layer_embedding_layouts(
        test_embeddings_by_layer,
        test_samples,
        task,
    )
    if layer_indices != test_layer_indices:
        raise InvalidConfigError(
            "train/test layer sets differ for learnable weighting: "
            f"{layer_indices} vs {test_layer_indices}"
        )
    if train_num_options != test_num_options or train_feature_shape != test_feature_shape:
        raise InvalidConfigError(
            "train/test feature layouts differ for learnable weighting: "
            f"{train_feature_shape} vs {test_feature_shape}"
        )
    batch_size = int(probe_cfg.training.batch_size)
    eval_batch_size = int(probe_cfg.training.get("eval_batch_size", batch_size))
    train_idx = torch.arange(train_labels.size(0), dtype=torch.long)
    test_idx = torch.arange(test_labels.size(0), dtype=torch.long)
    train_loader = _make_loader(
        train_embeddings_by_layer,
        train_labels,
        train_idx,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_options=train_num_options,
    )
    test_loader = _make_loader(
        test_embeddings_by_layer,
        test_labels,
        test_idx,
        batch_size=eval_batch_size,
        shuffle=False,
        seed=seed,
        num_options=test_num_options,
    )
    probe = _make_layer_weighted_probe(
        task=task,
        input_dim=train_feature_shape[-1],
        num_layers=len(layer_indices),
        probe_cfg=probe_cfg,
        mixed_feature_rank=len(train_feature_shape),
    )
    result = train_probe_from_loaders(
        train_loader=train_loader,
        val_loader=test_loader,
        test_loader=test_loader,
        probe=probe,
        task=task,
        config=probe_cfg,
        seed=seed,
        device=device,
    )
    row = _probe_result_row(
        LEARNABLE_WEIGHTING_LAYER_INDEX,
        seed,
        result,
        task,
        layer_aggregation=LEARNABLE_WEIGHTING_AGGREGATION,
    )
    row["split_strategy"] = "official_test"
    return _annotate_learnable_row(row, layer_indices, _remap_layer_weights(probe, layer_indices))


def evaluate_clustering_layer(
    layer_index: int,
    embedding: torch.Tensor,
    samples: list[DatasetSample],
    task: ClusteringTask,
    seed: int,
    max_samples: int,
) -> dict[str, float | int | str]:
    """Evaluate one layer with bounded KMeans instead of supervised probe training."""
    x, y = task.build_features(samples, embedding.float())
    if x.size(0) < task.num_clusters:
        raise InvalidConfigError(
            f"clustering needs at least {task.num_clusters} samples, got {x.size(0)}"
        )
    if max_samples > 0 and x.size(0) > max_samples:
        generator = torch.Generator().manual_seed(seed)
        idx = torch.randperm(x.size(0), generator=generator)[:max_samples]
        x = x[idx]
        y = y[idx]

    assignments = task.cluster(x, seed=seed)
    metrics = task.evaluate(assignments, y)
    primary = float(metrics[task.spec.metric_primary])
    return {
        "layer_index": layer_index,
        "seed": seed,
        "primary_metric": primary,
        "secondary_metrics_json": str(
            {k: v for k, v in metrics.items() if k != task.spec.metric_primary}
        ),
        "ci_lower": primary,
        "ci_upper": primary,
        "num_params": 0,
        "val_primary": primary,
        "layer_aggregation": SINGLE_LAYER_AGGREGATION,
    }


def _materialize_samples(
    dataset_spec,
    split: str,
    limit: int | None,
    task_type: TaskType,
) -> list[DatasetSample]:
    """Pull samples for one split from HF, limited to ``limit`` if set."""
    return list(
        iter_hf_samples(
            dataset_spec,
            split=split,
            limit=limit,
            task_type=task_type,
            log_prefix="[03_probes]",
        )
    )


def _infer_num_classes(samples: list[DatasetSample]) -> int:
    labels = {str(sample.label) for sample in samples if sample.label is not None}
    if len(labels) < 2:
        raise InvalidConfigError(f"expected at least 2 distinct labels, found {len(labels)}")
    return len(labels)


def _clustering_eval_limit() -> int:
    raw = os.environ.get("CLUSTERING_EVAL_LIMIT")
    if raw is None:
        return DEFAULT_CLUSTERING_EVAL_LIMIT
    try:
        return int(raw)
    except ValueError as exc:
        raise InvalidConfigError(f"invalid CLUSTERING_EVAL_LIMIT={raw!r}") from exc


def _subsample_limit(cfg: DictConfig, split: str) -> int | None:
    subsample = cfg.dataset.get("subsample", {})
    if split == cfg.dataset.get("split_train"):
        return subsample.get("train")
    if split == cfg.dataset.get("split_val"):
        return subsample.get("val")
    if split == cfg.dataset.get("split_test"):
        return subsample.get("test")
    return subsample.get(split)


def _layer_indices(cfg: DictConfig, num_layers: int) -> range:
    start = 0 if bool(cfg.extraction.get("include_embedding_layer", False)) else 1
    return range(start, num_layers + 1)


def _embedding_key(
    cfg: DictConfig,
    model_spec,
    dataset_spec,
    split: str,
    layer_index: int,
) -> EmbeddingKey:
    sample_order = (
        "length_sorted"
        if bool(cfg.extraction.get("sort_by_length", False))
        else "dataset"
    )
    return EmbeddingKey(
        model_id=model_spec.hf_id,
        dataset_id=dataset_spec.hf_id,
        dataset_subset=dataset_spec.hf_subset,
        split=split,
        layer_index=layer_index,
        pooling_name=normalize_pooling_name(str(cfg.pooling.name), model_spec=model_spec),
        dtype=str(cfg.cache.embeddings_dtype),
        extraction_max_length=int(cfg.extraction.max_length),
        truncation=bool(cfg.extraction.truncation),
        sample_order=sample_order,
        representation_side=representation_side_for_model(model_spec),
    )


def _official_test_requested(cfg: DictConfig) -> bool:
    evaluation = cfg.probe.get("evaluation", {})
    return str(evaluation.get("strategy", "auto")) in {"auto", "official_test"}


def _official_test_split(cfg: DictConfig) -> str:
    split = cfg.dataset.get("split_test")
    if not split:
        raise InvalidConfigError(
            "probe.evaluation.strategy=official_test requires dataset.split_test"
        )
    return str(split)


def _split_available(dataset_spec, split: str) -> bool:
    return split in set(dataset_spec.splits_available or [])


def _official_test_required(cfg: DictConfig) -> bool:
    evaluation = cfg.probe.get("evaluation", {})
    return str(evaluation.get("strategy", "auto")) == "official_test"


def _labels_are_usable_for_official_test(
    samples: list[DatasetSample],
    task_type: TaskType,
) -> bool:
    if task_type == TaskType.CLUSTERING:
        return False
    labels = [sample.label for sample in samples]
    if not labels or any(label is None for label in labels):
        return False
    unique = {str(label) for label in labels}
    if len(unique) < 2:
        return False
    if unique == {"-1"}:
        return False
    return True


def _learnable_weighting_enabled(cfg: DictConfig, task: Task, model_spec=None) -> bool:
    if task.spec.task_type not in TRAINABLE_LAYER_WEIGHTING_TASKS:
        return False
    if str(cfg.probe.type) not in TRAINABLE_LAYER_WEIGHTING_PROBES:
        return False
    task_default_pooling = cfg.task.get("pooling_default")
    if not task_default_pooling:
        return False
    configured_pooling = str(cfg.pooling.name)
    default_pooling = str(task_default_pooling)
    if model_spec is not None:
        configured_pooling = normalize_pooling_name(
            configured_pooling,
            model_spec=model_spec,
        )
        default_pooling = normalize_pooling_name(default_pooling, model_spec=model_spec)
    return configured_pooling == default_pooling


def _learnable_probe_key(
    base_embedding_key: EmbeddingKey,
    task: Task,
    probe_cfg: DictConfig,
    probe_hash: str,
    seed: int,
) -> ProbeKey:
    sentinel_embedding_key = replace(
        base_embedding_key,
        layer_index=LEARNABLE_WEIGHTING_LAYER_INDEX,
    )
    return ProbeKey(
        embedding_key=sentinel_embedding_key,
        task_name=task.spec.task_type.value,
        probe_type=str(probe_cfg.type),
        seed=int(seed),
        probe_config_hash=probe_hash,
    )


def _load_embeddings_by_layer(
    cache: CacheStore,
    keys: dict[int, EmbeddingKey],
    *,
    mmap: bool = False,
) -> dict[int, torch.Tensor]:
    loader = cache.load_embedding_mmap if mmap else cache.load_embedding
    return {int(layer): loader(key) for layer, key in sorted(keys.items())}


def _attach_probe_metadata(
    row: dict[str, float | int | str],
    key: EmbeddingKey,
    task: Task,
    probe_cfg: DictConfig,
) -> dict[str, float | int | str]:
    row["model_id"] = key.model_id
    row["dataset_id"] = key.dataset_id
    row["task_type"] = task.spec.task_type.value
    row["pooling_name"] = key.pooling_name
    row["representation_side"] = key.representation_side
    row["probe_type"] = str(probe_cfg.type)
    row["split_strategy"] = row.get(
        "split_strategy",
        "random_60_20_20",
    )
    return row


def run_probes(
    cfg: DictConfig,
    cache: CacheStore,
    embedding_keys: dict[int, EmbeddingKey],
    samples: list[DatasetSample],
    task: Task,
    test_embedding_keys: dict[int, EmbeddingKey] | None = None,
    test_samples: list[DatasetSample] | None = None,
    model_spec=None,
) -> list[dict[str, float | int | str]]:
    """Iterate seeds and layers, train each probe, append rows to the cache.

    Returns the list of rows written. Already-cached probe rows are skipped and
    reloaded from the cache.
    """
    from tqdm.auto import tqdm

    rows: list[dict[str, float | int | str]] = []
    seeds = list(cfg.probe.get("seeds", [int(cfg.seed)]))
    probe_cfg = cfg.probe
    training_device = cfg.get("device", "cpu")
    use_official_test = test_embedding_keys is not None and test_samples is not None
    logger.info("probe training device=%s", training_device)
    if use_official_test:
        logger.info("probe evaluation strategy=official_test")
    probe_hash = stable_hash(OmegaConf.to_container(probe_cfg, resolve=True))
    pending_probe_writes: list[tuple[ProbeKey, dict[str, float | int | str]]] = []
    clustering_eval_limit = (
        _clustering_eval_limit() if task.spec.task_type == TaskType.CLUSTERING else 0
    )
    lazy_supported = _lazy_probe_supported(task)
    layer_iter = tqdm(
        embedding_keys.items(),
        total=len(embedding_keys),
        desc=f"probe ({len(seeds)} seed{'s' if len(seeds) > 1 else ''}/layer)",
        unit="layer",
    )
    for layer_index, key in layer_iter:
        if isinstance(task, ClusteringTask) or not lazy_supported:
            embedding = cache.load_embedding(key)
        else:
            embedding = cache.load_embedding_mmap(key)
        test_embedding = None
        if use_official_test:
            test_key = test_embedding_keys.get(layer_index) if test_embedding_keys else None
            if test_key is None:
                raise InvalidConfigError(f"missing test embedding key for layer {layer_index}")
            test_embedding = (
                cache.load_embedding(test_key)
                if not lazy_supported
                else cache.load_embedding_mmap(test_key)
            )
        for seed in seeds:
            probe_key = ProbeKey(
                embedding_key=key,
                task_name=task.spec.task_type.value,
                probe_type=str(probe_cfg.type),
                seed=int(seed),
                probe_config_hash=probe_hash,
            )
            if cache.has_probe(probe_key):
                rows.append(cache.load_probe(probe_key))
                continue
            if isinstance(task, ClusteringTask):
                if use_official_test:
                    raise InvalidConfigError(
                        "official_test evaluation is not supported for clustering"
                    )
                row = evaluate_clustering_layer(
                    layer_index=layer_index,
                    embedding=embedding,
                    samples=samples,
                    task=task,
                    seed=int(seed),
                    max_samples=clustering_eval_limit,
                )
            elif (
                lazy_supported
                and use_official_test
                and test_embedding is not None
                and test_samples is not None
            ):
                row = train_one_layer_official_test_batched(
                    layer_index=layer_index,
                    train_embedding=embedding,
                    train_samples=samples,
                    test_embedding=test_embedding,
                    test_samples=test_samples,
                    task=task,
                    probe_cfg=probe_cfg,
                    seed=int(seed),
                    device=training_device,
                )
            elif use_official_test and test_embedding is not None and test_samples is not None:
                row = train_one_layer_official_test(
                    layer_index=layer_index,
                    train_embedding=embedding,
                    train_samples=samples,
                    test_embedding=test_embedding,
                    test_samples=test_samples,
                    task=task,
                    probe_cfg=probe_cfg,
                    seed=int(seed),
                    device=training_device,
                )
            elif lazy_supported:
                row = train_one_layer_batched(
                    layer_index=layer_index,
                    embedding=embedding,
                    samples=samples,
                    task=task,
                    probe_cfg=probe_cfg,
                    seed=int(seed),
                    device=training_device,
                )
            else:
                row = train_one_layer(
                    layer_index=layer_index,
                    embedding=embedding,
                    samples=samples,
                    task=task,
                    probe_cfg=probe_cfg,
                    seed=int(seed),
                    device=training_device,
                )
            row = _attach_probe_metadata(row, key, task, probe_cfg)
            pending_probe_writes.append((probe_key, row))
            rows.append(row)
    if _learnable_weighting_enabled(cfg, task, model_spec=model_spec):
        base_key = embedding_keys[sorted(embedding_keys)[0]]
        train_embeddings_by_layer: dict[int, torch.Tensor] | None = None
        test_embeddings_by_layer: dict[int, torch.Tensor] | None = None
        for seed in seeds:
            probe_key = _learnable_probe_key(
                base_embedding_key=base_key,
                task=task,
                probe_cfg=probe_cfg,
                probe_hash=probe_hash,
                seed=int(seed),
            )
            if cache.has_probe(probe_key):
                rows.append(cache.load_probe(probe_key))
                continue
            if train_embeddings_by_layer is None:
                train_embeddings_by_layer = _load_embeddings_by_layer(
                    cache,
                    embedding_keys,
                    mmap=lazy_supported,
                )
            if use_official_test:
                if test_embedding_keys is None or test_samples is None:
                    raise InvalidConfigError("missing official-test inputs for learnable weighting")
                if test_embeddings_by_layer is None:
                    test_embeddings_by_layer = _load_embeddings_by_layer(
                        cache,
                        test_embedding_keys,
                        mmap=lazy_supported,
                    )
                if lazy_supported:
                    row = train_learnable_layer_weighting_official_test_batched(
                        train_embeddings_by_layer=train_embeddings_by_layer,
                        train_samples=samples,
                        test_embeddings_by_layer=test_embeddings_by_layer,
                        test_samples=test_samples,
                        task=task,
                        probe_cfg=probe_cfg,
                        seed=int(seed),
                        device=training_device,
                    )
                else:
                    row = train_learnable_layer_weighting_official_test(
                        train_embeddings_by_layer=train_embeddings_by_layer,
                        train_samples=samples,
                        test_embeddings_by_layer=test_embeddings_by_layer,
                        test_samples=test_samples,
                        task=task,
                        probe_cfg=probe_cfg,
                        seed=int(seed),
                        device=training_device,
                    )
            else:
                if lazy_supported:
                    row = train_learnable_layer_weighting_batched(
                        embeddings_by_layer=train_embeddings_by_layer,
                        samples=samples,
                        task=task,
                        probe_cfg=probe_cfg,
                        seed=int(seed),
                        device=training_device,
                    )
                else:
                    row = train_learnable_layer_weighting(
                        embeddings_by_layer=train_embeddings_by_layer,
                        samples=samples,
                        task=task,
                        probe_cfg=probe_cfg,
                        seed=int(seed),
                        device=training_device,
                    )
            row = _attach_probe_metadata(row, base_key, task, probe_cfg)
            pending_probe_writes.append((probe_key, row))
            rows.append(row)
    cache.save_probe_many(pending_probe_writes)
    return rows


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train probes for every layer of (model, dataset, task)."""
    model_spec = get_model_spec(cfg.model.name)
    dataset_spec = get_dataset_spec(cfg.dataset.name)
    task_type = TaskType(cfg.task.name)
    check_compatibility(model_spec, task_type)
    if bool(cfg.extraction.get("sort_by_length", False)):
        raise InvalidConfigError(
            "Probe training requires extraction.sort_by_length=false so cached "
            "embeddings stay aligned with labels."
        )

    task = build_task(task_type, num_classes=dataset_spec.num_classes)
    cache = CacheStore(Path(cfg.cache.root))

    split = resolve_train_split(cfg.dataset, dataset_spec)
    limit = _subsample_limit(cfg, split)
    samples = _materialize_samples(
        dataset_spec,
        split=split,
        limit=limit,
        task_type=task_type,
    )
    if task_type == TaskType.CLUSTERING and dataset_spec.num_classes is None:
        task = ClusteringTask(num_clusters=_infer_num_classes(samples))
    logger.info(
        "training probes on %d samples for %s / %s / %s",
        len(samples),
        model_spec.name,
        dataset_spec.name,
        task_type.value,
    )

    test_split: str | None = None
    test_samples: list[DatasetSample] | None = None
    if _official_test_requested(cfg) and task_type != TaskType.CLUSTERING:
        try:
            candidate_split = _official_test_split(cfg)
        except InvalidConfigError:
            if _official_test_required(cfg):
                raise
            candidate_split = None
        if candidate_split == split:
            if _official_test_required(cfg):
                raise InvalidConfigError(
                    "official_test would use the same split for train and test; "
                    "use probe.evaluation.strategy=auto or random_split instead"
                )
            candidate_split = None
        if candidate_split is not None and not _split_available(dataset_spec, candidate_split):
            if _official_test_required(cfg):
                raise InvalidConfigError(
                    f"official test split {candidate_split!r} is not available for "
                    f"dataset={dataset_spec.name}"
                )
            logger.info(
                "official test split %s is not available; "
                "falling back to random train split evaluation",
                candidate_split,
            )
            candidate_split = None
        if candidate_split is not None:
            try:
                candidate_samples = _materialize_samples(
                    dataset_spec,
                    split=candidate_split,
                    limit=_subsample_limit(cfg, candidate_split),
                    task_type=task_type,
                )
            except Exception as exc:  # noqa: BLE001
                if _official_test_required(cfg):
                    raise InvalidConfigError(
                        f"could not materialize official test split "
                        f"{candidate_split!r}: {exc}"
                    ) from exc
                logger.info(
                    "official test split %s could not be materialized (%s); "
                    "falling back to random train split evaluation",
                    candidate_split,
                    exc,
                )
                candidate_samples = []
            if _labels_are_usable_for_official_test(candidate_samples, task_type):
                test_split = candidate_split
                test_samples = candidate_samples
                logger.info(
                    "evaluating probes on %d samples from official split=%s",
                    len(test_samples),
                    test_split,
                )
            elif _official_test_required(cfg):
                raise InvalidConfigError(
                    f"official test split {candidate_split!r} is missing usable labels"
                )
            else:
                logger.info(
                    "official test split %s is missing usable labels; "
                    "falling back to random train split evaluation",
                    candidate_split,
                )

    embedding_keys = {
        layer_index: _embedding_key(cfg, model_spec, dataset_spec, split, layer_index)
        for layer_index in _layer_indices(cfg, model_spec.num_layers)
        if cache.has_embedding(
            _embedding_key(cfg, model_spec, dataset_spec, split, layer_index)
        )
    }
    if not embedding_keys:
        raise InvalidConfigError(
            "no cached embeddings found; run pipeline/01_extract_embeddings.py first"
        )

    test_embedding_keys: dict[int, EmbeddingKey] | None = None
    if test_split is not None:
        test_embedding_keys = {
            layer_index: _embedding_key(cfg, model_spec, dataset_spec, test_split, layer_index)
            for layer_index in embedding_keys
        }
        missing = [
            layer_index
            for layer_index, key in test_embedding_keys.items()
            if not cache.has_embedding(key)
        ]
        if missing:
            preview = ", ".join(str(layer) for layer in missing[:5])
            if _official_test_required(cfg):
                raise InvalidConfigError(
                    f"missing official-test embeddings for split={test_split}, "
                    f"layer(s): {preview}. Run 01_extract_embeddings.py with "
                    f"extraction.split={test_split} first."
                )
            logger.info(
                "missing official-test embeddings for split=%s layer(s) %s; "
                "falling back to random train split evaluation",
                test_split,
                preview,
            )
            test_embedding_keys = None
            test_samples = None

    rows = run_probes(
        cfg,
        cache,
        embedding_keys,
        samples,
        task,
        test_embedding_keys=test_embedding_keys,
        test_samples=test_samples,
        model_spec=model_spec,
    )
    logger.info("wrote %d probe rows to %s", len(rows), cache.root)
    # Surface the schema-level details for debugging without re-querying parquet.
    if rows:
        logger.debug("first row: %s", rows[0])
    _ = asdict  # silence linter for unused-but-handy import


if __name__ == "__main__":
    main()
