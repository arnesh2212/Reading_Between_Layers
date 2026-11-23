"""Sweep resolver: expand sentinels, validate tuples, emit a manifest.

This is the mandatory gate before any multirun. Pure config walking plus
registry lookups; never loads a model, dataset, or probe.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reading_between_layers.caching.keys import EmbeddingKey  # noqa: E402
from reading_between_layers.caching.store import CacheStore  # noqa: E402
from reading_between_layers.datasets.registry import (  # noqa: E402
    DatasetSpec,
    get_dataset_spec,
    list_enabled_datasets,
)
from reading_between_layers.datasets.splits import resolve_train_split  # noqa: E402
from reading_between_layers.exceptions import InvalidConfigError, UnsupportedModelTaskCombo  # noqa: E402
from reading_between_layers.models.registry import (  # noqa: E402
    ModelSpec,
    get_model_spec,
    list_models_by_tier,
)
from reading_between_layers.pooling.selection import (  # noqa: E402
    normalize_pooling_names,
    representation_side_for_model,
)
from reading_between_layers.tasks.compatibility import TaskType, check_compatibility  # noqa: E402
from reading_between_layers.utils.manifest import Manifest, write_manifest  # noqa: E402

SENTINEL_ALL_TIER_A = "__all_tier_a__"
SENTINEL_ALL_COMPATIBLE = "__all_compatible__"
SENTINEL_AUTO_FROM_DATASET = "__auto_from_dataset__"
SENTINEL_TASK_POOLING = "__task_default_plus_alternatives__"

BATCH_SIZE_FOR_VRAM = 32
VRAM_SAFETY_FACTOR = 100
VRAM_BUDGET_BYTES = 20 * 1024**3
BYTES_PER_FP16 = 2
BYTES_PER_DTYPE = {
    "float16": 2,
    "float32": 4,
    "bfloat16": 2,
}
DEFAULT_SAMPLE_CAP = 10_000
CONFIG_DIR = REPO_ROOT / "configs"


def estimate_vram_bytes(spec: ModelSpec, batch_size: int = BATCH_SIZE_FOR_VRAM) -> int:
    """Heuristic VRAM estimate: hidden * num_layers * batch * 4 bytes, scaled by safety factor."""
    base = spec.hidden_size * spec.num_layers * batch_size * 4
    return base * VRAM_SAFETY_FACTOR


def estimate_cache_bytes(
    model_spec: ModelSpec,
    dataset_spec: DatasetSpec,
    bytes_per_scalar: int = BYTES_PER_FP16,
    include_embedding_layer: bool = False,
) -> int:
    """Estimate per-layer cache bytes summed across the model's extracted layers."""
    n = min(dataset_spec.approx_size_total, DEFAULT_SAMPLE_CAP)
    layer_count = model_spec.num_layers + (1 if include_embedding_layer else 0)
    return model_spec.hidden_size * n * bytes_per_scalar * layer_count


def _to_python(value: Any) -> Any:
    """Convert an OmegaConf container to a plain Python value; pass strings through."""
    if isinstance(value, str):
        return value
    return OmegaConf.to_object(value) if OmegaConf.is_config(value) else value


def _materialize_models(value: Any) -> list[ModelSpec]:
    if isinstance(value, str):
        if value == SENTINEL_ALL_TIER_A:
            return list_models_by_tier("A")
        raise ValueError(f"unknown model sentinel: {value}")
    return [get_model_spec(name) for name in value]


def _materialize_datasets(value: Any) -> list[DatasetSpec]:
    if isinstance(value, str):
        if value == SENTINEL_ALL_COMPATIBLE:
            return list_enabled_datasets()
        raise ValueError(f"unknown dataset sentinel: {value}")
    return [get_dataset_spec(name) for name in value]


def _datasets_for_model(
    model_spec: ModelSpec,
    datasets: list[DatasetSpec],
    datasets_by_family: Any,
) -> list[DatasetSpec]:
    """Return the dataset subset configured for a model family, if any."""
    if not datasets_by_family:
        return datasets
    policy = _to_python(datasets_by_family)
    if not isinstance(policy, dict):
        raise ValueError("experiment.datasets_by_family must be a mapping")
    names = policy.get(model_spec.family)
    if names is None:
        return datasets
    by_name = {dataset.name: dataset for dataset in datasets}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise ValueError(
            f"datasets_by_family[{model_spec.family}] references datasets outside "
            f"experiment.datasets: {missing}"
        )
    return [by_name[name] for name in names]


def _materialize_tasks(value: Any, dataset_spec: DatasetSpec) -> list[TaskType]:
    if isinstance(value, str):
        if value == SENTINEL_AUTO_FROM_DATASET:
            return list(dataset_spec.task_types)
        return [TaskType(value)]
    return [TaskType(t) for t in value]


def _materialize_poolings(value: Any, task_cfg: dict[str, Any]) -> list[str]:
    if isinstance(value, str):
        if value == SENTINEL_TASK_POOLING:
            opts = list(task_cfg.get("pooling_options") or [])
            default = task_cfg.get("pooling_default")
            if default and default not in opts:
                opts = [default] + opts
            return opts
        return [value]
    return list(value)


def _load_task_config(task_name: str) -> dict[str, Any]:
    raw = OmegaConf.load(CONFIG_DIR / "task" / f"{task_name}.yaml")
    return OmegaConf.to_object(raw)


def _load_dataset_config(dataset_name: str) -> dict[str, Any]:
    raw = OmegaConf.load(CONFIG_DIR / "dataset" / f"{dataset_name}.yaml")
    return OmegaConf.to_object(raw)


def _first_extracted_layer(cfg: DictConfig) -> int:
    return 0 if bool(cfg.extraction.get("include_embedding_layer", False)) else 1


def _cache_sample_order(cfg: DictConfig) -> str:
    return "length_sorted" if bool(cfg.extraction.get("sort_by_length", False)) else "dataset"


def _cache_bytes_per_scalar(cfg: DictConfig) -> int:
    return BYTES_PER_DTYPE.get(str(cfg.cache.embeddings_dtype), BYTES_PER_FP16)


def _check_stale_entries(
    verified_assets: dict[str, Any], max_age_days: int, warnings: list[str]
) -> None:
    """Append a warning per stale or non-ok verification entry."""
    from datetime import timedelta as _timedelta

    threshold = datetime.now(UTC) - _timedelta(days=max_age_days)
    for kind in ("models", "datasets"):
        for name, record in verified_assets.get(kind, {}).items():
            verified_at = record.get("verified_at")
            if not verified_at:
                continue
            try:
                ts = datetime.fromisoformat(verified_at)
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts < threshold:
                warnings.append(
                    f"{kind}/{name}: verification entry older than "
                    f"{max_age_days} days (last verified {verified_at})"
                )


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=REPO_ROOT,
        )
        return out.decode().strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _resolver_settings(exp: DictConfig) -> tuple[bool, int]:
    """Extract resolver knobs from an experiment config, falling back to defaults."""
    resolver_cfg = exp.get("resolver") if hasattr(exp, "get") else None
    strict = False
    max_age = 7
    if resolver_cfg is not None:
        strict = bool(OmegaConf.select(resolver_cfg, "strict_verification", default=False))
        max_age = int(OmegaConf.select(resolver_cfg, "max_verified_age_days", default=7))
    return strict, max_age


def _verification_status(
    verified_assets: dict[str, Any] | None, kind: str, name: str
) -> dict[str, Any] | None:
    if verified_assets is None:
        return None
    return verified_assets.get(kind, {}).get(name)


def resolve_sweep(
    cfg: DictConfig,
    cache_root: Path | None = None,
    experiment_name: str | None = None,
    verified_assets: dict[str, Any] | None = None,
) -> Manifest:
    """Expand the experiment sentinels into a Manifest of validated tuples."""
    cache_root = Path(cache_root) if cache_root else Path(cfg.cache.root)
    cache = CacheStore(cache_root)

    exp = cfg.experiment
    strict_verification, max_verified_age_days = _resolver_settings(exp)
    seeds = list(exp.seeds)
    probes_in = _to_python(exp.probes)
    probes = list(probes_in) if probes_in else [str(cfg.probe.type)]

    models = _materialize_models(_to_python(exp.models))
    datasets = _materialize_datasets(_to_python(exp.datasets))
    datasets_by_family = exp.get("datasets_by_family")
    tasks_in = _to_python(exp.tasks)
    poolings_in = _to_python(exp.poolings)

    tuples: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_quant: set[str] = set()
    seen_extraction: set[tuple[str, str, str]] = set()
    seen_verification_warning: set[tuple[str, str]] = set()
    cached_skipped = 0
    total_cache_bytes = 0
    verification_failures: list[str] = []

    def _record_verification(kind: str, name: str) -> None:
        status_record = _verification_status(verified_assets, kind, name)
        if status_record is None:
            return
        status = status_record.get("status")
        if status == "ok":
            return
        key_pair = (kind, name)
        if key_pair in seen_verification_warning:
            return
        seen_verification_warning.add(key_pair)
        error = status_record.get("error", "")
        msg = f"{kind}/{name}: verification status={status}"
        if error:
            msg = f"{msg} ({error})"
        warnings.append(msg)
        verification_failures.append(msg)

    for model_spec in models:
        if model_spec.size_tier == "B" and model_spec.name not in seen_quant:
            warnings.append(f"{model_spec.name}: requires int8, will use bitsandbytes")
            seen_quant.add(model_spec.name)
        vram = estimate_vram_bytes(model_spec)
        if vram > VRAM_BUDGET_BYTES:
            warnings.append(
                f"{model_spec.name}: VRAM estimate {vram} exceeds budget "
                f"{VRAM_BUDGET_BYTES}; will skip"
            )
            continue
        _record_verification("models", model_spec.name)

        for dataset_spec in _datasets_for_model(model_spec, datasets, datasets_by_family):
            dataset_cfg = _load_dataset_config(dataset_spec.name)
            _record_verification("datasets", dataset_spec.name)
            for task_type in _materialize_tasks(tasks_in, dataset_spec):
                try:
                    check_compatibility(model_spec, task_type)
                except UnsupportedModelTaskCombo:
                    continue
                task_cfg = _load_task_config(task_type.value)
                allowed_probes = task_cfg.get("probe_types_allowed") or []
                pooling_options = task_cfg.get("pooling_options") or []
                raw_poolings = _materialize_poolings(poolings_in, task_cfg)
                poolings = normalize_pooling_names(
                    (pooling for pooling in raw_poolings if pooling in pooling_options),
                    model_spec=model_spec,
                )
                for probe in probes:
                    if probe not in allowed_probes:
                        continue
                    for pooling in poolings:
                        for seed in seeds:
                            split = resolve_train_split(dataset_cfg, dataset_spec)
                            key = EmbeddingKey(
                                model_id=model_spec.hf_id,
                                dataset_id=dataset_spec.hf_id,
                                dataset_subset=dataset_spec.hf_subset,
                                split=split,
                                layer_index=_first_extracted_layer(cfg),
                                pooling_name=pooling,
                                dtype=str(cfg.cache.embeddings_dtype),
                                extraction_max_length=int(cfg.extraction.max_length),
                                truncation=bool(cfg.extraction.truncation),
                                sample_order=_cache_sample_order(cfg),
                                representation_side=representation_side_for_model(model_spec),
                            )
                            if cache.has_embedding(key):
                                cached_skipped += 1
                                continue
                            tuples.append(
                                {
                                    "model": model_spec.name,
                                    "dataset": dataset_spec.name,
                                    "task": task_type.value,
                                    "probe": probe,
                                    "pooling": pooling,
                                    "seed": int(seed),
                                    "vram_estimate_bytes": int(vram),
                                }
                            )
                            extraction_key = (
                                model_spec.name,
                                dataset_spec.name,
                                pooling,
                            )
                            if extraction_key not in seen_extraction:
                                seen_extraction.add(extraction_key)
                                total_cache_bytes += estimate_cache_bytes(
                                    model_spec,
                                    dataset_spec,
                                    bytes_per_scalar=_cache_bytes_per_scalar(cfg),
                                    include_embedding_layer=bool(
                                        cfg.extraction.get("include_embedding_layer", False)
                                    ),
                                )

    if cached_skipped > 0:
        warnings.append(f"{cached_skipped} tuples already cached, will skip")

    if verification_failures and strict_verification:
        raise InvalidConfigError(
            "strict_verification=true and some assets are not ok:\n"
            + "\n".join(verification_failures)
        )

    if verified_assets is not None:
        _check_stale_entries(verified_assets, max_verified_age_days, warnings)

    now = datetime.now(UTC)
    run_id = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{os.urandom(3).hex()}"
    commands = [
        "python pipeline/01_extract_embeddings.py --multirun  # set MANIFEST_HASH=<hash> first",
        "python pipeline/02_compute_geometry.py --multirun  # set MANIFEST_HASH=<hash> first",
        "python pipeline/03_train_probes.py --multirun  # set MANIFEST_HASH=<hash> first",
        "python pipeline/04_export_results.py --cache-root outputs/cache --out results/paper",
    ]
    return Manifest(
        run_id=run_id,
        timestamp=now.isoformat(),
        git_commit=_git_commit(),
        experiment=experiment_name or str(exp.mode),
        tuples=tuples,
        estimates={
            "total_tuples": len(tuples),
            "estimated_cache_delta_bytes": int(total_cache_bytes),
            "cached_skipped_tuples": cached_skipped,
        },
        warnings=warnings,
        commands=commands,
    )


def summarize(manifest: Manifest) -> str:
    """Return a human-readable summary in the format from docs/spec/17_cli_and_sweep.md."""
    models = sorted({t["model"] for t in manifest.tuples})
    datasets = sorted({t["dataset"] for t in manifest.tuples})
    tasks = sorted({t["task"] for t in manifest.tuples})
    probes = sorted({t["probe"] for t in manifest.tuples})
    seeds = sorted({t["seed"] for t in manifest.tuples})
    cache_gb = manifest.estimates.get("estimated_cache_delta_bytes", 0) / (1024**3)
    lines = [
        f"Sweep resolution: experiment={manifest.experiment}",
        f"Models: {len(models)}",
        f"Datasets: {len(datasets)} compatible",
        f"Tasks: {len(tasks)} task types",
        f"Probes: {probes}",
        f"Seeds: {len(seeds)}",
        "",
        f"Total tuples: {len(manifest.tuples):,}",
        f"Estimated cache delta: {cache_gb:.1f} GB",
    ]
    if manifest.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in manifest.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)


def _dispatch(manifest: Manifest, manifest_path: Path) -> None:
    env = os.environ.copy()
    env["MANIFEST_HASH"] = manifest.hash()
    for cmd in manifest.commands:
        print(f"Dispatching: {cmd}")
        subprocess.run(cmd, shell=True, env=env, cwd=REPO_ROOT, check=False)


VERIFIED_ASSETS_PATH = REPO_ROOT / "outputs" / "verified_assets.json"


def _load_verified_assets() -> dict[str, Any] | None:
    """Read verified_assets.json. Return None if absent; caller decides what to do."""
    if not VERIFIED_ASSETS_PATH.exists():
        return None
    import json as _json

    return _json.loads(VERIFIED_ASSETS_PATH.read_text())


@hydra.main(version_base=None, config_path="../configs", config_name="resolver_config")
def main(cfg: DictConfig) -> None:
    """Resolve a sweep, validate tuples, and emit a manifest."""
    dry_run = bool(cfg.get("dry_run", False))
    dispatch = bool(cfg.get("dispatch", False))

    verified_assets = _load_verified_assets()
    if verified_assets is None:
        print(
            f"Note: {VERIFIED_ASSETS_PATH.relative_to(REPO_ROOT)} not found; "
            "Hugging Face asset verification is skipped."
        )

    try:
        exp_name = HydraConfig.get().runtime.choices.get("experiment")
    except ValueError:
        exp_name = None
    manifest = resolve_sweep(
        cfg, experiment_name=exp_name, verified_assets=verified_assets
    )
    print(summarize(manifest))

    if dry_run:
        return

    runs_dir = REPO_ROOT / "outputs" / "runs" / manifest.run_id
    path = write_manifest(manifest, runs_dir)
    print(f"\nManifest: {path.relative_to(REPO_ROOT)}")
    print(f"Hash: {manifest.hash()}")

    if dispatch:
        _dispatch(manifest, path)


if __name__ == "__main__":
    main()
