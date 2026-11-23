"""Hydra entry point: compute enabled geometry metrics over cached embeddings."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
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

from reading_between_layers.caching.keys import EmbeddingKey, GeometryKey  # noqa: E402
from reading_between_layers.caching.store import CacheStore  # noqa: E402
from reading_between_layers.datasets.registry import get_dataset_spec  # noqa: E402
from reading_between_layers.datasets.splits import resolve_train_split  # noqa: E402
from reading_between_layers.geometry import compute_all_geometry, resolve_geometry_sample_size  # noqa: E402
from reading_between_layers.geometry.registry import GEOMETRY_REGISTRY  # noqa: E402
from reading_between_layers.models.registry import get_model_spec  # noqa: E402
from reading_between_layers.pooling.selection import (  # noqa: E402
    normalize_pooling_name,
    representation_side_for_model,
)
from reading_between_layers.utils.hashing import stable_hash  # noqa: E402
from reading_between_layers.utils.tabular import write_csv_sidecar  # noqa: E402

logger = logging.getLogger(__name__)

_PAIRWISE_REFERENCE_METRICS = {"linear_cka"}


def _resolve_geometry_device(cfg: DictConfig) -> torch.device:
    """Use the configured device for torch-native metrics when available."""
    requested = cfg.get("device", "cpu") if hasattr(cfg, "get") else "cpu"
    device = torch.device(str(requested))
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("geometry requested cuda, but CUDA is unavailable; using CPU")
        return torch.device("cpu")
    return device


def _to_geometry_device(
    embedding,
    labels,
    sample_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """Slice on CPU first, then move the bounded geometry sample to ``device``."""
    cpu_embedding = embedding[:sample_size].detach().cpu()
    if torch.is_tensor(labels):
        cpu_labels = labels[:sample_size].detach().cpu()
    elif labels is not None:
        cpu_labels = labels[:sample_size]
    else:
        cpu_labels = None
    if device.type == "cpu":
        return cpu_embedding, cpu_labels, cpu_embedding, cpu_labels
    try:
        gpu_embedding = cpu_embedding.to(device, non_blocking=True)
        gpu_labels = (
            cpu_labels.to(device, non_blocking=True) if torch.is_tensor(cpu_labels) else cpu_labels
        )
        return gpu_embedding, gpu_labels, cpu_embedding, cpu_labels
    except torch.cuda.OutOfMemoryError:
        logger.warning(
            "could not move geometry sample with %d rows to %s; using CPU for this layer",
            sample_size,
            device,
        )
        torch.cuda.empty_cache()
        return cpu_embedding, cpu_labels, cpu_embedding, cpu_labels


def _compute_metrics_with_fallback(
    metric_names: list[str],
    embedding: torch.Tensor,
    labels: torch.Tensor | None,
    cpu_embedding: torch.Tensor,
    cpu_labels: torch.Tensor | None,
    metrics_cfg: DictConfig,
) -> tuple[dict[str, float], str]:
    """Compute metrics on the preferred device, falling back to CPU after CUDA OOM."""
    try:
        return compute_all_geometry(embedding, labels, metrics_cfg), str(embedding.device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if (
            not isinstance(exc, torch.cuda.OutOfMemoryError)
            and "CUDA out of memory" not in str(exc)
        ):
            raise
        logger.warning(
            "geometry metric batch (%s) ran out of CUDA memory; retrying on CPU",
            ", ".join(metric_names),
        )
        torch.cuda.empty_cache()
        cpu_output = compute_all_geometry(cpu_embedding, cpu_labels, metrics_cfg)
        return cpu_output, str(cpu_embedding.device)


def _to_reference_device(
    reference_embedding,
    sample_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if reference_embedding is None:
        return None, None
    cpu_reference = reference_embedding[:sample_size].detach().cpu()
    if device.type == "cpu":
        return cpu_reference, cpu_reference
    try:
        gpu_reference = cpu_reference.to(device, non_blocking=True)
        return gpu_reference, cpu_reference
    except torch.cuda.OutOfMemoryError:
        logger.warning("could not move reference geometry sample to %s; using CPU", device)
        torch.cuda.empty_cache()
        return cpu_reference, cpu_reference


def _compute_pairwise_metric_with_fallback(
    metric_name: str,
    embedding: torch.Tensor,
    reference_embedding: torch.Tensor | None,
    cpu_embedding: torch.Tensor,
    cpu_reference_embedding: torch.Tensor | None,
) -> tuple[dict[str, float], str, str | None]:
    cls = GEOMETRY_REGISTRY.get(metric_name)
    output_keys = _metric_output_keys(metric_name)
    if cls is None:
        return {key: float("nan") for key in output_keys}, str(embedding.device), "metric_missing"
    if reference_embedding is None:
        return {key: float("nan") for key in output_keys}, str(embedding.device), "missing_reference"
    try:
        output = cls().compute(embedding, labels=reference_embedding)
        return output, str(embedding.device), None
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if (
            not isinstance(exc, torch.cuda.OutOfMemoryError)
            and "CUDA out of memory" not in str(exc)
        ):
            raise
        if cpu_reference_embedding is None:
            return {key: float("nan") for key in output_keys}, str(cpu_embedding.device), (
                "missing_reference"
            )
        logger.warning("metric %s ran out of CUDA memory; retrying on CPU", metric_name)
        torch.cuda.empty_cache()
        output = cls().compute(cpu_embedding, labels=cpu_reference_embedding)
        return output, str(cpu_embedding.device), None


def _runnable_metrics(
    metrics_cfg: dict[str, bool],
    model=None,
    log_skips: bool = False,
) -> dict[str, bool]:
    """Filter enabled metrics to those runnable from cached embeddings."""
    runnable: dict[str, bool] = {}
    skipped: list[str] = []
    for metric_name, enabled in metrics_cfg.items():
        if not enabled:
            runnable[metric_name] = enabled
            continue
        cls = GEOMETRY_REGISTRY.get(metric_name)
        requires_jacobian = bool(getattr(cls, "requires_jacobian", False)) if cls else False
        if requires_jacobian and model is None:
            skipped.append(metric_name)
            runnable[metric_name] = False
            continue
        runnable[metric_name] = enabled
    if skipped and log_skips:
        logger.info(
            "skipping geometry metric(s) requiring a live model in cached-embedding mode: %s",
            ", ".join(skipped),
        )
    return runnable


def _metric_output_keys(metric_name: str) -> list[str]:
    cls = GEOMETRY_REGISTRY.get(metric_name)
    return list(getattr(cls, "output_keys", [metric_name])) if cls else [metric_name]


def _layer_indices(cfg: DictConfig, num_layers: int) -> range:
    start = 0 if bool(cfg.extraction.get("include_embedding_layer", False)) else 1
    return range(start, num_layers + 1)


def _sample_order(cfg: DictConfig) -> str:
    return "length_sorted" if bool(cfg.extraction.get("sort_by_length", False)) else "dataset"


def geometry_rows_for_layer(
    embedding_key: EmbeddingKey,
    embedding,
    labels,
    cfg: DictConfig,
    cache: CacheStore,
    model_id: str,
    dataset_id: str,
    device: torch.device | None = None,
    deferred_cache_items: list[tuple[GeometryKey, dict[str, float]]] | None = None,
    reference_embedding=None,
    reference_layer_index: int | None = None,
) -> list[dict]:
    """Compute every enabled metric for one cached embedding and emit parquet rows.

    Already-cached metric/layer combinations are skipped. Failed metrics are
    materialized as rows with ``failure_reason`` populated and ``metric_value``
    set to ``NaN``.
    """
    metrics_cfg = _runnable_metrics(dict(cfg.geometry.metrics))
    params_hash = stable_hash(OmegaConf.to_container(cfg.geometry, resolve=True))
    rows: list[dict] = []
    now = dt.datetime.now(dt.UTC).isoformat()
    geometry_input = str(cfg.geometry.get("geometry_input", "pooled"))
    dtype = str(cfg.cache.embeddings_dtype) if hasattr(cfg, "cache") else "float16"
    sample_size = resolve_geometry_sample_size(int(embedding.shape[0]), cfg.geometry)
    sample_fraction = cfg.geometry.get("sample_fraction", 1.0)
    geometry_device = device or _resolve_geometry_device(cfg)
    geometry_embedding, geometry_labels, cpu_embedding, cpu_labels = _to_geometry_device(
        embedding=embedding,
        labels=labels,
        sample_size=sample_size,
        device=geometry_device,
    )
    reference_on_device, cpu_reference = _to_reference_device(
        reference_embedding=reference_embedding,
        sample_size=sample_size,
        device=geometry_device,
    )

    pending: list[tuple[str, GeometryKey]] = []
    pending_pairwise: list[tuple[str, GeometryKey]] = []
    for metric_name, enabled in metrics_cfg.items():
        if not enabled:
            continue
        geom_key = GeometryKey(
            embedding_key=embedding_key,
            metric_name=metric_name,
            metric_params_hash=params_hash,
        )
        if cache.has_geometry(geom_key):
            continue
        if metric_name in _PAIRWISE_REFERENCE_METRICS:
            pending_pairwise.append((metric_name, geom_key))
        else:
            pending.append((metric_name, geom_key))
    if not pending and not pending_pairwise:
        return rows

    cache_items: list[tuple[GeometryKey, dict[str, float]]] = []
    if pending:
        pending_names = [metric_name for metric_name, _ in pending]
        batch_cfg = OmegaConf.create(
            {
                "metrics": {metric_name: True for metric_name in pending_names},
                "sample_size": None,
                "sample_fraction": 1.0,
            }
        )
        try:
            output, compute_device = _compute_metrics_with_fallback(
                metric_names=pending_names,
                embedding=geometry_embedding,
                labels=geometry_labels,
                cpu_embedding=cpu_embedding,
                cpu_labels=cpu_labels,
                metrics_cfg=batch_cfg,
            )
            batch_failure: str | None = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("geometry metric batch failed: %s", exc)
            output = {}
            compute_device = str(geometry_embedding.device)
            batch_failure = f"{type(exc).__name__}: {exc}"

        for metric_name, geom_key in pending:
            payload = {
                key: float(output.get(key, float("nan")))
                for key in _metric_output_keys(metric_name)
            }
            cache_items.append((geom_key, payload))
            for key, value in payload.items():
                rows.append(
                    {
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "split": embedding_key.split,
                        "layer_index": embedding_key.layer_index,
                        "pooling_name": embedding_key.pooling_name,
                        "representation_side": embedding_key.representation_side,
                        "metric_name": key,
                        "metric_value": float(value),
                        "metric_params_hash": params_hash,
                        "sample_size": sample_size,
                        "sample_fraction": float(sample_fraction),
                        "compute_device": compute_device,
                        "computed_at": now,
                        "dtype": dtype,
                        "geometry_input": geometry_input,
                        "failure_reason": batch_failure,
                    }
                )

    for metric_name, geom_key in pending_pairwise:
        try:
            output, compute_device, failure = _compute_pairwise_metric_with_fallback(
                metric_name=metric_name,
                embedding=geometry_embedding,
                reference_embedding=reference_on_device,
                cpu_embedding=cpu_embedding,
                cpu_reference_embedding=cpu_reference,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("metric %s failed: %s", metric_name, exc)
            output = {}
            compute_device = str(geometry_embedding.device)
            failure = f"{type(exc).__name__}: {exc}"
        payload = {
            key: float(output.get(key, float("nan")))
            for key in _metric_output_keys(metric_name)
        }
        cache_items.append((geom_key, payload))
        for key, value in payload.items():
            rows.append(
                {
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "split": embedding_key.split,
                    "layer_index": embedding_key.layer_index,
                    "pooling_name": embedding_key.pooling_name,
                    "representation_side": embedding_key.representation_side,
                    "metric_name": key,
                    "metric_value": float(value),
                    "metric_params_hash": params_hash,
                    "sample_size": sample_size,
                    "sample_fraction": float(sample_fraction),
                    "compute_device": compute_device,
                    "computed_at": now,
                    "dtype": dtype,
                    "geometry_input": geometry_input,
                    "failure_reason": failure,
                    "reference_layer_index": reference_layer_index,
                    "reference_pooling_name": embedding_key.pooling_name,
                }
            )
    if deferred_cache_items is None:
        cache.save_geometry_many(cache_items)
    else:
        deferred_cache_items.extend(cache_items)
    return rows


def write_geometry_parquet(rows: list[dict], path: Path) -> Path:
    """Append rows to a long-form geometry sidecar at ``path``; mkdir as needed."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists():
        try:
            previous = pd.read_parquet(path)
            df = pd.concat([previous, df], ignore_index=True, sort=False)
            dedupe_cols = [
                col
                for col in (
                    "model_id",
                    "dataset_id",
                    "split",
                    "layer_index",
                    "pooling_name",
                    "representation_side",
                    "metric_name",
                    "metric_params_hash",
                )
                if col in df.columns
            ]
            if dedupe_cols:
                df = df.drop_duplicates(subset=dedupe_cols, keep="last")
        except (OSError, ValueError) as exc:
            logger.warning("could not merge existing geometry parquet %s: %s", path, exc)
    df.to_parquet(path, compression="zstd", index=False)
    write_csv_sidecar(df, path)
    return path


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compute every enabled geometry metric for the cached (model, dataset, layer) set."""
    from tqdm.auto import tqdm

    print(
        f"[02_geometry] model={cfg.model.name} dataset={cfg.dataset.name} "
        f"sample_size={cfg.geometry.get('sample_size')} "
        f"sample_fraction={cfg.geometry.get('sample_fraction', 1.0)}"
    )
    model_spec = get_model_spec(cfg.model.name)
    dataset_spec = get_dataset_spec(cfg.dataset.name)
    pooling_name = normalize_pooling_name(str(cfg.pooling.name), model_spec=model_spec)
    representation_side = representation_side_for_model(model_spec)
    cache = CacheStore(Path(cfg.cache.root))
    geometry_device = _resolve_geometry_device(cfg)
    print(f"[02_geometry] torch metrics device={geometry_device}")

    enabled_metrics = [
        k
        for k, v in _runnable_metrics(dict(cfg.geometry.metrics), log_skips=True).items()
        if v
    ]
    print(f"[02_geometry] enabled metrics ({len(enabled_metrics)}): {enabled_metrics}")

    split = resolve_train_split(cfg.dataset, dataset_spec)
    accumulated: list[dict] = []
    deferred_cache_items: list[tuple[GeometryKey, dict[str, float]]] = []
    previous_embedding = None
    previous_layer_index: int | None = None
    layer_iter = tqdm(
        _layer_indices(cfg, model_spec.num_layers),
        desc="geometry layers",
        unit="layer",
    )
    for layer_index in layer_iter:
        key = EmbeddingKey(
            model_id=model_spec.hf_id,
            dataset_id=dataset_spec.hf_id,
            dataset_subset=dataset_spec.hf_subset,
            split=split,
            layer_index=layer_index,
            pooling_name=pooling_name,
            dtype=str(cfg.cache.embeddings_dtype),
            extraction_max_length=int(cfg.extraction.max_length),
            truncation=bool(cfg.extraction.truncation),
            sample_order=_sample_order(cfg),
            representation_side=representation_side,
        )
        if not cache.has_embedding(key):
            previous_embedding = None
            previous_layer_index = None
            continue
        embedding = cache.load_embedding(key)
        rows = geometry_rows_for_layer(
            embedding_key=key,
            embedding=embedding,
            labels=None,
            cfg=cfg,
            cache=cache,
            model_id=model_spec.hf_id,
            dataset_id=dataset_spec.hf_id,
            device=geometry_device,
            deferred_cache_items=deferred_cache_items,
            reference_embedding=previous_embedding,
            reference_layer_index=previous_layer_index,
        )
        accumulated.extend(rows)
        previous_embedding = embedding
        previous_layer_index = layer_index

    if not accumulated:
        logger.info("no new geometry rows to write; cache is up to date")
        return
    cache.save_geometry_many(deferred_cache_items)
    out_path = Path(cfg.cache.root) / "geometry" / "geometry.parquet"
    written = write_geometry_parquet(accumulated, out_path)
    logger.info("wrote %d geometry rows to %s", len(accumulated), written)
    # Emit a JSON sidecar of the in-memory rows for quick inspection.
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps(accumulated, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
