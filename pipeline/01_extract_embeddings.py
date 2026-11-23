"""Hydra entry point: extract layerwise pooled embeddings for one (model, dataset)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reading_between_layers.exceptions import InvalidConfigError  # noqa: E402
from reading_between_layers.utils.logging_config import configure_logging  # noqa: E402
from reading_between_layers.utils.manifest import find_latest_manifest_hash  # noqa: E402

configure_logging()


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
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from reading_between_layers.caching.keys import EmbeddingKey  # noqa: E402
from reading_between_layers.caching.store import CacheStore  # noqa: E402
from reading_between_layers.datasets.materialize import iter_hf_samples  # noqa: E402
from reading_between_layers.datasets.registry import get_dataset_spec  # noqa: E402
from reading_between_layers.datasets.splits import resolve_train_split  # noqa: E402
from reading_between_layers.extraction.layerwise import (  # noqa: E402
    extract_all_layer_embeddings,
    extract_multi_pooling_layer_embeddings,
)
from reading_between_layers.models.loader import load_model  # noqa: E402
from reading_between_layers.models.registry import get_model_spec  # noqa: E402
from reading_between_layers.pooling import get_pooling  # noqa: E402
from reading_between_layers.pooling.selection import (  # noqa: E402
    normalize_pooling_names,
    representation_side_for_model,
)
from reading_between_layers.tasks.compatibility import TaskType, check_compatibility  # noqa: E402

logger = logging.getLogger(__name__)


def _resolve_extraction_split(cfg: DictConfig, dataset_spec) -> str:
    configured = cfg.extraction.get("split")
    if configured:
        return str(configured)
    return resolve_train_split(cfg.dataset, dataset_spec)


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


def _sample_order(cfg: DictConfig) -> str:
    return "length_sorted" if bool(cfg.extraction.get("sort_by_length", False)) else "dataset"


def _requested_pooling_names(cfg: DictConfig, model_spec) -> list[str]:
    """Resolve one or more poolings for this extraction process.

    ``cfg.pooling.name`` stays as the primary Hydra config group so existing
    single-pooling invocations keep working. A runner can add
    ``extraction.poolings=[cls,mean,...]`` to share the model forward pass.
    """
    names = [str(cfg.pooling.name)]
    configured = cfg.extraction.get("poolings")
    if configured is not None:
        if isinstance(configured, str):
            extra = [item.strip() for item in configured.split(",")]
        elif OmegaConf.is_config(configured):
            extra = [str(item) for item in OmegaConf.to_container(configured, resolve=True)]
        else:
            extra = [str(item) for item in configured]
        for name in extra:
            if name and name not in names:
                names.append(name)
    return normalize_pooling_names(names, model_spec=model_spec)


def _embedding_keys_for_pooling(
    cfg: DictConfig,
    model_spec,
    dataset_spec,
    split: str,
    pooling_name: str,
    num_layers: int,
) -> dict[int, EmbeddingKey]:
    return {
        layer_index: EmbeddingKey(
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
            representation_side=representation_side_for_model(model_spec),
        )
        for layer_index in _layer_indices(cfg, num_layers)
    }


def _init_wandb(cfg: DictConfig) -> None:
    if not bool(cfg.get("wandb", {}).get("enabled", False)):
        return
    try:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            tags=list(cfg.wandb.get("tags", [])),
            config=dict(cfg),
            reinit=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("wandb init failed (%s); continuing without it", exc)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run extraction for the configured (model, dataset, task, pooling) tuple."""
    model_spec = get_model_spec(cfg.model.name)
    pooling_names = _requested_pooling_names(cfg, model_spec)
    print(
        f"[01_extract] model={cfg.model.name} dataset={cfg.dataset.name} "
        f"task={cfg.task.name} poolings={pooling_names}"
    )
    dataset_spec = get_dataset_spec(cfg.dataset.name)
    task_type = TaskType(cfg.task.name)
    check_compatibility(model_spec, task_type)

    _init_wandb(cfg)

    cache = CacheStore(Path(cfg.cache.root))
    split = _resolve_extraction_split(cfg, dataset_spec)
    planned_keys = {
        pooling_name: _embedding_keys_for_pooling(
            cfg=cfg,
            model_spec=model_spec,
            dataset_spec=dataset_spec,
            split=split,
            pooling_name=pooling_name,
            num_layers=model_spec.num_layers,
        )
        for pooling_name in pooling_names
    }
    if all(
        cache.has_embedding(key)
        for keys in planned_keys.values()
        for key in keys.values()
    ):
        logger.info(
            "all %d pooling/layer embedding group(s) cached; skipping model load",
            sum(len(keys) for keys in planned_keys.values()),
        )
        return

    print(f"[01_extract] loading poolings={pooling_names}")
    poolings = {
        pooling_name: get_pooling(
            pooling_name,
            **({"hidden_size": model_spec.hidden_size} if pooling_name == "attention" else {}),
        )
        for pooling_name in pooling_names
    }
    print(f"[01_extract] loading model {model_spec.hf_id} (dtype={cfg.dtype})")
    wrapper = load_model(
        model_spec, dtype=str(cfg.dtype), quantization=cfg.model.get("quantization")
    )
    representation_side = representation_side_for_model(model_spec)
    num_layers = wrapper.num_layers(side=representation_side or "auto")
    print(f"[01_extract] model loaded, num_layers={num_layers}")

    embedding_keys_by_pooling = {
        pooling_name: _embedding_keys_for_pooling(
            cfg=cfg,
            model_spec=model_spec,
            dataset_spec=dataset_spec,
            split=split,
            pooling_name=pooling_name,
            num_layers=num_layers,
        )
        for pooling_name in pooling_names
    }
    if all(
        cache.has_embedding(key)
        for keys in embedding_keys_by_pooling.values()
        for key in keys.values()
    ):
        logger.info("all pooling/layer embedding groups cached after model load")
        return

    limit = _subsample_limit(cfg, split)
    samples = list(
        iter_hf_samples(
            dataset_spec,
            split=split,
            limit=limit,
            task_type=task_type,
            log_prefix="[load]",
        )
    )
    logger.info(
        "extracting %d samples from %s split=%s for %s",
        len(samples),
        dataset_spec.name,
        split,
        model_spec.name,
    )
    if not samples:
        raise InvalidConfigError(
            f"no usable samples for dataset={dataset_spec.name!r} split={split!r} "
            f"task={task_type.value!r}; choose a labeled split for supervised extraction"
        )

    if len(pooling_names) == 1:
        pooling_name = pooling_names[0]
        embedding_keys = embedding_keys_by_pooling[pooling_name]
        extract_all_layer_embeddings(
            model=wrapper,
            samples=samples,
            pooling=poolings[pooling_name],
            config=cfg,
            cache=cache,
            embedding_keys=embedding_keys,
            return_embeddings=False,
        )
    else:
        extract_multi_pooling_layer_embeddings(
            model=wrapper,
            samples=samples,
            poolings=poolings,
            config=cfg,
            cache=cache,
            embedding_keys_by_pooling=embedding_keys_by_pooling,
            return_embeddings=False,
        )
    logger.info(
        "extracted %d pooling/layer group(s), cache root=%s",
        sum(len(keys) for keys in embedding_keys_by_pooling.values()),
        cfg.cache.root,
    )


if __name__ == "__main__":
    main()
