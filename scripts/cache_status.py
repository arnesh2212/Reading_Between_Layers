#!/usr/bin/env python
"""Check whether planned local-sweep work is already materialized in cache."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reading_between_layers.caching.keys import EmbeddingKey, GeometryKey, ProbeKey  # noqa: E402
from reading_between_layers.datasets.registry import get_dataset_spec  # noqa: E402
from reading_between_layers.datasets.splits import resolve_train_split  # noqa: E402
from reading_between_layers.models.registry import get_model_spec  # noqa: E402
from reading_between_layers.pooling.selection import (  # noqa: E402
    normalize_pooling_name,
    representation_side_for_model,
)
from reading_between_layers.probing.layer_weighting import (  # noqa: E402
    LEARNABLE_WEIGHTING_LAYER_INDEX,
    TRAINABLE_LAYER_WEIGHTING_PROBES,
    TRAINABLE_LAYER_WEIGHTING_TASKS,
)
from reading_between_layers.tasks.compatibility import TaskType  # noqa: E402
from reading_between_layers.utils.hashing import stable_hash  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    group = subparsers.add_parser(
        "group-complete",
        help="exit 0 if downstream geometry/probe results for a model/dataset are complete",
    )
    group.add_argument("--plan-dir", required=True)
    group.add_argument("--cache-root", required=True)
    group.add_argument("--geometry-config", required=True)
    group.add_argument("--cache-embeddings-dtype", default=None)
    group.add_argument("--model", required=True)
    group.add_argument("--dataset", required=True)
    group.add_argument("--skip-geometry", action="store_true")
    group.add_argument("--skip-probes", action="store_true")
    group.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _manifest_hashes(cache_root: Path, kind: str) -> set[str]:
    path = cache_root / "manifest.json"
    if not path.exists():
        return set()
    hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("kind") == kind and record.get("hash"):
                hashes.add(str(record["hash"]))
    return hashes


def _rows(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.reader(handle, delimiter="\t") if row]


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=overrides)


def _cache_overrides(args: argparse.Namespace) -> list[str]:
    overrides = [f"cache.root={args.cache_root}"]
    if args.cache_embeddings_dtype:
        overrides.append(f"cache.embeddings_dtype={args.cache_embeddings_dtype}")
    return overrides


def _layer_indices(cfg, num_layers: int) -> range:
    start = 0 if bool(cfg.extraction.get("include_embedding_layer", False)) else 1
    return range(start, num_layers + 1)


def _embedding_key(cfg, layer_index: int) -> EmbeddingKey:
    model_spec = get_model_spec(cfg.model.name)
    dataset_spec = get_dataset_spec(cfg.dataset.name)
    split = resolve_train_split(cfg.dataset, dataset_spec)
    sample_order = (
        "length_sorted"
        if bool(cfg.extraction.get("sort_by_length", False))
        else "dataset"
    )
    return EmbeddingKey(
        model_id=str(cfg.model.hf_id),
        dataset_id=str(cfg.dataset.hf_id),
        dataset_subset=cfg.dataset.hf_subset,
        split=split,
        layer_index=layer_index,
        pooling_name=normalize_pooling_name(str(cfg.pooling.name), model_spec=model_spec),
        dtype=str(cfg.cache.embeddings_dtype),
        extraction_max_length=int(cfg.extraction.max_length),
        truncation=bool(cfg.extraction.truncation),
        sample_order=sample_order,
        representation_side=representation_side_for_model(model_spec),
    )


def _expected_geometry_hashes(args: argparse.Namespace) -> set[str]:
    if args.skip_geometry:
        return set()
    expected: set[str] = set()
    rows = _rows(Path(args.plan_dir) / "geometry_jobs.tsv")
    for model, dataset, task, pooling in rows:
        if model != args.model or dataset != args.dataset:
            continue
        cfg = _compose(
            _cache_overrides(args)
            + [
                f"model={model}",
                f"dataset={dataset}",
                f"task={task}",
                f"pooling={pooling}",
                f"geometry={args.geometry_config}",
            ]
        )
        params_hash = stable_hash(OmegaConf.to_container(cfg.geometry, resolve=True))
        enabled_metrics = [name for name, enabled in dict(cfg.geometry.metrics).items() if enabled]
        model_spec = get_model_spec(model)
        for layer_index in _layer_indices(cfg, model_spec.num_layers):
            embedding_key = _embedding_key(cfg, layer_index)
            for metric_name in enabled_metrics:
                expected.add(
                    GeometryKey(
                        embedding_key=embedding_key,
                        metric_name=metric_name,
                        metric_params_hash=params_hash,
                    ).hash()
                )
    return expected


def _expected_probe_hashes(args: argparse.Namespace) -> set[str]:
    if args.skip_probes:
        return set()
    expected: set[str] = set()
    rows = _rows(Path(args.plan_dir) / "probe_jobs.tsv")
    for model, dataset, task, probe, pooling, seeds_raw in rows:
        if model != args.model or dataset != args.dataset:
            continue
        cfg = _compose(
            _cache_overrides(args)
            + [
                f"model={model}",
                f"dataset={dataset}",
                f"task={task}",
                f"probe={probe}",
                f"pooling={pooling}",
                f"probe.seeds={seeds_raw}",
            ]
        )
        seeds = [int(seed) for seed in json.loads(seeds_raw)]
        probe_hash = stable_hash(OmegaConf.to_container(cfg.probe, resolve=True))
        model_spec = get_model_spec(model)
        for layer_index in _layer_indices(cfg, model_spec.num_layers):
            embedding_key = _embedding_key(cfg, layer_index)
            for seed in seeds:
                expected.add(
                    ProbeKey(
                        embedding_key=embedding_key,
                        task_name=task,
                        probe_type=probe,
                        seed=seed,
                        probe_config_hash=probe_hash,
                    ).hash()
                )
        task_type = TaskType(task)
        normalized_pooling = normalize_pooling_name(
            str(cfg.pooling.name),
            model_spec=model_spec,
        )
        normalized_default = normalize_pooling_name(
            str(cfg.task.get("pooling_default")),
            model_spec=model_spec,
        )
        if (
            task_type in TRAINABLE_LAYER_WEIGHTING_TASKS
            and probe in TRAINABLE_LAYER_WEIGHTING_PROBES
            and normalized_pooling == normalized_default
        ):
            sentinel_embedding_key = _embedding_key(cfg, LEARNABLE_WEIGHTING_LAYER_INDEX)
            for seed in seeds:
                expected.add(
                    ProbeKey(
                        embedding_key=sentinel_embedding_key,
                        task_name=task,
                        probe_type=probe,
                        seed=seed,
                        probe_config_hash=probe_hash,
                    ).hash()
                )
    return expected


def _group_complete(args: argparse.Namespace) -> int:
    if args.skip_geometry and args.skip_probes:
        if args.verbose:
            print("not complete: no downstream stages enabled")
        return 1

    cache_root = Path(args.cache_root)
    expected_geometry = _expected_geometry_hashes(args)
    expected_probes = _expected_probe_hashes(args)
    if not expected_geometry and not expected_probes:
        if args.verbose:
            print("not complete: no expected downstream rows in plan")
        return 1

    present_geometry = _manifest_hashes(cache_root, "geometry")
    present_probes = _manifest_hashes(cache_root, "probe")
    missing_geometry = expected_geometry - present_geometry
    missing_probes = expected_probes - present_probes
    if args.verbose:
        print(
            "geometry "
            f"{len(expected_geometry) - len(missing_geometry)}/{len(expected_geometry)}, "
            "probes "
            f"{len(expected_probes) - len(missing_probes)}/{len(expected_probes)}"
        )
    return 0 if not missing_geometry and not missing_probes else 1


def main() -> None:
    args = _parse_args()
    if args.command == "group-complete":
        raise SystemExit(_group_complete(args))
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
