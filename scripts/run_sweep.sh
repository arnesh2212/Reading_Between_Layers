#!/usr/bin/env bash
#
# Run the layerwise sweep end to end for one experiment config.
#
# Expands configs/experiment/${EXPERIMENT}.yaml into concrete jobs and runs, per
# model/dataset combination:
#   01 extraction -> 02 geometry -> 03 probes
# and finally exports the cache into results/ (pipeline/04_export_results.py).
#
# Useful knobs (environment variables):
#   EXPERIMENT=paper                 # paper | paper_extra | scaling_encoders | scaling_mamba | ...
#   GEOMETRY_CONFIG=fast_only        # geometry metrics used in the paper
#   OUTPUT_ROOT=outputs
#   CACHE_ROOT=outputs/cache
#   RESULTS_DIR=results/<experiment> # where the exported tables are written
#   DEVICE=cuda
#   DTYPE=float16                    # model load dtype
#   CACHE_EMBEDDINGS_DTYPE=float16   # saved probe-feature dtype
#   CONDA_ENV=<name>                 # optional: activate this conda env first
#   SKIP_EXTRACT=1 SKIP_GEOMETRY=1 SKIP_PROBES=1 SKIP_EXPORT=1
#   PLAN_ONLY=1                      # only write the job plan
#   CONTINUE_ON_ERROR=1
#   PRUNE_EMBEDDINGS_AFTER_GROUP=1   # delete cached embeddings once a combo is done

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENT="${EXPERIMENT:-paper}"
GEOMETRY_CONFIG="${GEOMETRY_CONFIG:-fast_only}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_ROOT}/cache}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
CACHE_EMBEDDINGS_DTYPE="${CACHE_EMBEDDINGS_DTYPE:-float16}"
CONDA_ENV="${CONDA_ENV:-}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT}}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SKIP_RESOLVE="${SKIP_RESOLVE:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SKIP_GEOMETRY="${SKIP_GEOMETRY:-0}"
SKIP_PROBES="${SKIP_PROBES:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
STREAM_BY_COMBO="${STREAM_BY_COMBO:-1}"
PRUNE_EMBEDDINGS_AFTER_GROUP="${PRUNE_EMBEDDINGS_AFTER_GROUP:-1}"
PRUNE_ON_FAILURE="${PRUNE_ON_FAILURE:-1}"
CLEAR_RUNTIME_CACHE_AFTER_GROUP="${CLEAR_RUNTIME_CACHE_AFTER_GROUP:-1}"

mkdir -p logs "$OUTPUT_ROOT/runs"

LOG_FILE="${LOG_FILE:-logs/sweep_${EXPERIMENT}.log}"
if [ "${NO_TEE:-0}" != "1" ]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

if [ -n "$CONDA_ENV" ]; then
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-1}"

RUN_ID="${EXPERIMENT}_$(date +%Y-%m-%d_%H-%M-%S)"
PLAN_DIR="${PLAN_DIR:-${OUTPUT_ROOT}/runs/${RUN_ID}}"
mkdir -p "$PLAN_DIR"

EXTRACT_TSV="$PLAN_DIR/extraction_jobs.tsv"
EXTRACT_GROUPED_TSV="$PLAN_DIR/extraction_grouped_jobs.tsv"
GEOMETRY_TSV="$PLAN_DIR/geometry_jobs.tsv"
PROBE_TSV="$PLAN_DIR/probe_jobs.tsv"
FAILURES_FILE="$PLAN_DIR/failures.tsv"
: > "$FAILURES_FILE"

echo ""
echo "=========="
echo "LAYERWISE SWEEP"
echo "Started: $(date -Iseconds)"
echo "Experiment: $EXPERIMENT"
echo "Geometry config: $GEOMETRY_CONFIG"
echo "Cache root: $CACHE_ROOT"
echo "Device: $DEVICE"
echo "DType: $DTYPE"
echo "Cache embeddings dtype: $CACHE_EMBEDDINGS_DTYPE"
echo "Plan dir: $PLAN_DIR"
echo "Stream by combo: $STREAM_BY_COMBO"
echo "Results dir: $RESULTS_DIR"
echo "Prune embeddings after group: $PRUNE_EMBEDDINGS_AFTER_GROUP"
echo "Prune embeddings after failed group: $PRUNE_ON_FAILURE"
echo "Clear runtime cache after group: $CLEAR_RUNTIME_CACHE_AFTER_GROUP"
echo "=========="

if [ "$SKIP_RESOLVE" != "1" ]; then
    echo ""
    echo "--- resolving manifest for provenance ---"
    python pipeline/00_resolve_sweep.py \
        "experiment=${EXPERIMENT}" \
        "cache.root=${CACHE_ROOT}" \
        "cache.embeddings_dtype=${CACHE_EMBEDDINGS_DTYPE}" \
        "extraction.sort_by_length=false"
else
    echo ""
    echo "--- resolver skipped: SKIP_RESOLVE=1 ---"
fi

echo ""
echo "--- building concrete local plan ---"
python - "$PLAN_DIR" "$EXPERIMENT" "$CACHE_ROOT" "$GEOMETRY_CONFIG" "$CACHE_EMBEDDINGS_DTYPE" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path.cwd()
CONFIG_DIR = REPO_ROOT / "configs"
plan_dir = Path(sys.argv[1])
experiment_name = sys.argv[2]
cache_root = sys.argv[3]
geometry_config = sys.argv[4]
cache_embeddings_dtype = sys.argv[5]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reading_between_layers.datasets.registry import get_dataset_spec, list_enabled_datasets  # noqa: E402
from reading_between_layers.exceptions import UnsupportedModelTaskCombo  # noqa: E402
from reading_between_layers.models.registry import get_model_spec, list_models_by_tier  # noqa: E402
from reading_between_layers.pooling.selection import normalize_pooling_names  # noqa: E402
from reading_between_layers.pooling.registry import POOLING_REGISTRY  # noqa: E402
from reading_between_layers.tasks.compatibility import TaskType, check_compatibility  # noqa: E402

SENTINEL_ALL_TIER_A = "__all_tier_a__"
SENTINEL_ALL_COMPATIBLE = "__all_compatible__"
SENTINEL_AUTO_FROM_DATASET = "__auto_from_dataset__"
SENTINEL_TASK_POOLING = "__task_default_plus_alternatives__"

BATCH_SIZE_FOR_VRAM = 32
VRAM_SAFETY_FACTOR = 100
VRAM_BUDGET_BYTES = 20 * 1024**3
SUPPORTED_PROBE_TASKS = {
    TaskType.CLASSIFICATION,
    TaskType.CLUSTERING,
    TaskType.MULTIPLE_CHOICE_QA,
    TaskType.PAIR_CLASSIFICATION,
    TaskType.STS,
}


def to_python(value):
    if isinstance(value, str):
        return value
    return OmegaConf.to_object(value) if OmegaConf.is_config(value) else value


def materialize_models(value):
    if isinstance(value, str):
        if value == SENTINEL_ALL_TIER_A:
            return list_models_by_tier("A")
        raise ValueError(f"unknown model sentinel: {value}")
    return [get_model_spec(name) for name in value]


def materialize_datasets(value):
    if isinstance(value, str):
        if value == SENTINEL_ALL_COMPATIBLE:
            return list_enabled_datasets()
        raise ValueError(f"unknown dataset sentinel: {value}")
    return [get_dataset_spec(name) for name in value]


def datasets_for_model(model_spec, datasets, datasets_by_family):
    """Return the dataset subset configured for a model family, if any."""
    if not datasets_by_family:
        return datasets
    policy = to_python(datasets_by_family)
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


def materialize_tasks(value, dataset_spec):
    if isinstance(value, str):
        if value == SENTINEL_AUTO_FROM_DATASET:
            return list(dataset_spec.task_types)
        return [TaskType(value)]
    return [TaskType(task) for task in value]


def materialize_poolings(value, task_cfg):
    if isinstance(value, str):
        if value == SENTINEL_TASK_POOLING:
            opts = list(task_cfg.get("pooling_options") or [])
            default = task_cfg.get("pooling_default")
            if default and default not in opts:
                opts = [default] + opts
            return opts
        return [value]
    return list(value)


def load_task_config(task_name: str) -> dict:
    raw = OmegaConf.load(CONFIG_DIR / "task" / f"{task_name}.yaml")
    return OmegaConf.to_object(raw)


def estimate_vram_bytes(spec) -> int:
    base = spec.hidden_size * spec.num_layers * BATCH_SIZE_FOR_VRAM * 4
    return base * VRAM_SAFETY_FACTOR


def load_dataset_config(dataset_name: str) -> dict:
    raw = OmegaConf.load(CONFIG_DIR / "dataset" / f"{dataset_name}.yaml")
    return OmegaConf.to_object(raw)


def resolve_plan_train_split(dataset_spec, dataset_cfg: dict) -> str:
    configured = dataset_cfg.get("split_train")
    if configured:
        return str(configured)
    if "train" in dataset_spec.splits_available:
        return "train"
    for key in ("split_val", "split_test"):
        configured = dataset_cfg.get(key)
        if configured:
            return str(configured)
    if dataset_spec.splits_available:
        return str(dataset_spec.splits_available[0])
    return "train"


def plan_test_split(dataset_spec, dataset_cfg: dict, task_type) -> str | None:
    if task_type == TaskType.CLUSTERING:
        return None
    split = dataset_cfg.get("split_test")
    if not split:
        return None
    split = str(split)
    train_split = resolve_plan_train_split(dataset_spec, dataset_cfg)
    if split == train_split:
        return None
    if split not in dataset_spec.splits_available:
        return None
    return split


def unsupported_local_task_reason(task_type, dataset_spec) -> str | None:
    if task_type not in SUPPORTED_PROBE_TASKS:
        return "probe runner does not support this task yet"
    if task_type == TaskType.CLUSTERING:
        if dataset_spec.label_field is None:
            return "local clustering requires labels"
    if task_type == TaskType.MULTIPLE_CHOICE_QA:
        if dataset_spec.label_field is None:
            return "local multiple-choice QA requires labels"
        if dataset_spec.num_classes is None:
            return "local multiple-choice QA requires a known option count"
    return None


with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
    cfg = compose(
        config_name="resolver_config",
        overrides=[
            f"experiment={experiment_name}",
            f"cache.root={cache_root}",
            f"cache.embeddings_dtype={cache_embeddings_dtype}",
            "extraction.sort_by_length=false",
            f"geometry={geometry_config}",
        ],
    )

exp = cfg.experiment
models = materialize_models(to_python(exp.models))
datasets = materialize_datasets(to_python(exp.datasets))
datasets_by_family = exp.get("datasets_by_family")
tasks_in = to_python(exp.tasks)
poolings_in = to_python(exp.poolings)
probes = list(to_python(exp.probes)) if to_python(exp.probes) else [str(cfg.probe.type)]
seeds = [int(seed) for seed in exp.seeds]
seeds_cli = "[" + ",".join(str(seed) for seed in seeds) + "]"

extraction_jobs: dict[tuple[str, str, str, str], tuple[str, str, str, str, str]] = {}
geometry_jobs: dict[tuple[str, str, str], tuple[str, str, str, str]] = {}
probe_jobs: set[tuple[str, str, str, str, str, str]] = set()
skipped: list[str] = []
pooling_config_names = {
    path.stem
    for path in (CONFIG_DIR / "pooling").glob("*.yaml")
    if not path.name.startswith("_")
}
supported_poolings = set(POOLING_REGISTRY) & pooling_config_names

for model_spec in models:
    vram = estimate_vram_bytes(model_spec)
    if vram > VRAM_BUDGET_BYTES:
        skipped.append(f"{model_spec.name}: VRAM estimate exceeds budget")
        continue
    for dataset_spec in datasets_for_model(model_spec, datasets, datasets_by_family):
        dataset_cfg = load_dataset_config(dataset_spec.name)
        for task_type in materialize_tasks(tasks_in, dataset_spec):
            unsupported_reason = unsupported_local_task_reason(task_type, dataset_spec)
            if unsupported_reason is not None:
                skipped.append(
                    f"{model_spec.name}/{dataset_spec.name}/{task_type.value}: "
                    f"{unsupported_reason}"
                )
                continue
            try:
                check_compatibility(model_spec, task_type)
            except UnsupportedModelTaskCombo as exc:
                skipped.append(
                    f"{model_spec.name}/{dataset_spec.name}/{task_type.value}: {exc}"
                )
                continue

            task_cfg = load_task_config(task_type.value)
            allowed_probes = set(task_cfg.get("probe_types_allowed") or [])
            valid_probes = [probe for probe in probes if probe in allowed_probes]
            if not valid_probes:
                skipped.append(
                    f"{model_spec.name}/{dataset_spec.name}/{task_type.value}: no valid probes"
                )
                continue

            pooling_options = set(task_cfg.get("pooling_options") or [])
            raw_poolings = [
                pooling
                for pooling in materialize_poolings(poolings_in, task_cfg)
                if pooling in pooling_options
            ]
            for pooling in normalize_pooling_names(raw_poolings, model_spec=model_spec):
                if pooling not in supported_poolings:
                    skipped.append(
                        f"{model_spec.name}/{dataset_spec.name}/{task_type.value}/{pooling}: "
                        "pooling is not runnable in this codebase"
                    )
                    continue

                base_row = (
                    model_spec.name,
                    dataset_spec.name,
                    task_type.value,
                    pooling,
                )
                train_split = resolve_plan_train_split(dataset_spec, dataset_cfg)
                test_split = plan_test_split(dataset_spec, dataset_cfg, task_type)
                extraction_jobs.setdefault(
                    (model_spec.name, dataset_spec.name, pooling, train_split),
                    base_row + (train_split,),
                )
                if test_split is not None:
                    extraction_jobs.setdefault(
                        (model_spec.name, dataset_spec.name, pooling, test_split),
                        base_row + (test_split,),
                    )
                key = (model_spec.name, dataset_spec.name, pooling)
                geometry_jobs.setdefault(key, base_row)

                for probe in valid_probes:
                    probe_jobs.add(
                        (
                            model_spec.name,
                            dataset_spec.name,
                            task_type.value,
                            probe,
                            pooling,
                            seeds_cli,
                        )
                    )


def write_tsv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(sorted(rows))


write_tsv(plan_dir / "extraction_jobs.tsv", extraction_jobs.values())
write_tsv(plan_dir / "geometry_jobs.tsv", geometry_jobs.values())
write_tsv(plan_dir / "probe_jobs.tsv", probe_jobs)

summary = {
    "experiment": experiment_name,
    "geometry_config": geometry_config,
    "cache_embeddings_dtype": cache_embeddings_dtype,
    "models": sorted({row[0] for row in extraction_jobs.values()}),
    "datasets": sorted({row[1] for row in extraction_jobs.values()}),
    "tasks": sorted({row[2] for row in probe_jobs}),
    "poolings": sorted({row[3] for row in extraction_jobs.values()}),
    "extraction_splits": sorted({row[4] for row in extraction_jobs.values()}),
    "probes": sorted({row[3] for row in probe_jobs}),
    "seeds": seeds,
    "extraction_jobs": len(extraction_jobs),
    "geometry_jobs": len(geometry_jobs),
    "probe_jobs": len(probe_jobs),
    "skipped_count": len(skipped),
    "skipped_examples": skipped[:50],
}
(plan_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(plan_dir / "summary.txt").write_text(
    "\n".join(
        [
            f"Experiment: {summary['experiment']}",
            f"Geometry config: {summary['geometry_config']}",
            f"Cache embeddings dtype: {summary['cache_embeddings_dtype']}",
            f"Models: {len(summary['models'])}",
            f"Datasets: {len(summary['datasets'])}",
            f"Tasks: {len(summary['tasks'])}",
            f"Poolings: {summary['poolings']}",
            f"Extraction splits: {summary['extraction_splits']}",
            f"Probes: {summary['probes']}",
            f"Seeds: {summary['seeds']}",
            f"Extraction jobs: {summary['extraction_jobs']}",
            f"Geometry jobs: {summary['geometry_jobs']}",
            f"Probe jobs: {summary['probe_jobs']}",
            f"Skipped combinations: {summary['skipped_count']}",
        ]
    )
    + "\n",
    encoding="utf-8",
)
PY

cat "$PLAN_DIR/summary.txt"

python - "$EXTRACT_TSV" "$EXTRACT_GROUPED_TSV" <<'PY'
from __future__ import annotations

import csv
import sys
from collections import OrderedDict
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
groups: dict[tuple[str, str, str, str], OrderedDict[str, None]] = {}
with source.open("r", newline="", encoding="utf-8") as handle:
    reader = csv.reader(handle, delimiter="\t")
    for row in reader:
        if len(row) < 5:
            continue
        model, dataset, task, pooling, split = row[:5]
        key = (model, dataset, task, split)
        groups.setdefault(key, OrderedDict())[pooling] = None

with target.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    for model, dataset, task, split in sorted(groups):
        writer.writerow([model, dataset, task, ",".join(groups[(model, dataset, task, split)]), split])
PY
echo "Grouped extraction jobs: $(wc -l < "$EXTRACT_GROUPED_TSV" | tr -d ' ')"

if [ "$PLAN_ONLY" = "1" ]; then
    echo "PLAN_ONLY=1, stopping after plan creation."
    exit 0
fi

COMMON_OVERRIDES=(
    "cache.root=${CACHE_ROOT}"
    "cache.embeddings_dtype=${CACHE_EMBEDDINGS_DTYPE}"
    "device=${DEVICE}"
    "dtype=${DTYPE}"
    "wandb.enabled=false"
    "extraction.sort_by_length=false"
)

record_failure() {
    local stage="$1"
    local label="$2"
    local rc="$3"
    printf '%s\t%s\t%s\n' "$stage" "$rc" "$label" >> "$FAILURES_FILE"
}

run_or_record() {
    local stage="$1"
    local label="$2"
    shift 2
    local rc=0
    "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then
        return 0
    fi
    echo "FAILED [$stage] rc=$rc: $label"
    record_failure "$stage" "$label" "$rc"
    if [ "$CONTINUE_ON_ERROR" = "1" ]; then
        return 0
    fi
    return "$rc"
}

count_rows() {
    wc -l < "$1" | tr -d ' '
}

failure_count() {
    wc -l < "$FAILURES_FILE" | tr -d ' '
}

run_extract_job() {
    local model="$1"
    local dataset="$2"
    local task="$3"
    local pooling="$4"
    local split="$5"
    local primary_pooling="$pooling"
    local pooling_overrides=()
    if [[ "$pooling" == *,* ]]; then
        primary_pooling="${pooling%%,*}"
        pooling_overrides=("+extraction.poolings=[${pooling}]")
    fi
    run_or_record "extract" "$model/$dataset/$task/$pooling/$split" \
        python pipeline/01_extract_embeddings.py \
            "${COMMON_OVERRIDES[@]}" \
            "model=${model}" \
            "dataset=${dataset}" \
            "task=${task}" \
            "pooling=${primary_pooling}" \
            "${pooling_overrides[@]}" \
            "extraction.split=${split}"
}

run_geometry_job() {
    local model="$1"
    local dataset="$2"
    local task="$3"
    local pooling="$4"
    run_or_record "geometry" "$model/$dataset/$task/$pooling/$GEOMETRY_CONFIG" \
        python pipeline/02_compute_geometry.py \
            "${COMMON_OVERRIDES[@]}" \
            "model=${model}" \
            "dataset=${dataset}" \
            "task=${task}" \
            "pooling=${pooling}" \
            "geometry=${GEOMETRY_CONFIG}"
}

run_probe_job() {
    local model="$1"
    local dataset="$2"
    local task="$3"
    local probe="$4"
    local pooling="$5"
    local seeds="$6"
    run_or_record "probe" "$model/$dataset/$task/$probe/$pooling/$seeds" \
        python pipeline/03_train_probes.py \
            "${COMMON_OVERRIDES[@]}" \
            "model=${model}" \
            "dataset=${dataset}" \
            "task=${task}" \
            "probe=${probe}" \
            "pooling=${pooling}" \
            "probe.seeds=${seeds}"
}

run_export() {
    if [ "$SKIP_EXPORT" = "1" ]; then
        return 0
    fi
    echo ""
    echo "--- exporting cache -> ${RESULTS_DIR} ---"
    run_or_record "export" "04_export_results" \
        python pipeline/04_export_results.py \
            "--cache-root=${CACHE_ROOT}" \
            "--out=${RESULTS_DIR}"
}

prune_group_embeddings() {
    local model="$1"
    local dataset="$2"
    if [ "$PRUNE_EMBEDDINGS_AFTER_GROUP" != "1" ]; then
        return 0
    fi
    if [ "$SKIP_GEOMETRY" = "1" ] || [ "$SKIP_PROBES" = "1" ]; then
        echo "[prune] skipped for $model/$dataset because geometry or probes are disabled"
        return 0
    fi
    echo "[prune] removing intermediate embeddings for $model/$dataset"
    run_or_record "prune" "$model/$dataset" \
        python scripts/prune_embedding_cache.py \
            "--cache-root=${CACHE_ROOT}" \
            "--model-name=${model}" \
            "--dataset-name=${dataset}" \
            --yes
}

clear_runtime_cache() {
    if [ "$CLEAR_RUNTIME_CACHE_AFTER_GROUP" != "1" ]; then
        return 0
    fi
    python scripts/clear_runtime_cache.py "--cache-root=${CACHE_ROOT}" || true
}

group_complete() {
    local model="$1"
    local dataset="$2"
    local args=(
        group-complete
        "--plan-dir=${PLAN_DIR}"
        "--cache-root=${CACHE_ROOT}"
        "--geometry-config=${GEOMETRY_CONFIG}"
        "--cache-embeddings-dtype=${CACHE_EMBEDDINGS_DTYPE}"
        "--model=${model}"
        "--dataset=${dataset}"
    )
    if [ "$SKIP_GEOMETRY" = "1" ]; then
        args+=(--skip-geometry)
    fi
    if [ "$SKIP_PROBES" = "1" ]; then
        args+=(--skip-probes)
    fi
    python scripts/cache_status.py "${args[@]}"
}

run_staged_pipeline() {
if [ "$SKIP_EXTRACT" = "1" ]; then
    echo ""
    echo "--- extraction skipped: SKIP_EXTRACT=1 ---"
else
    total="$(count_rows "$EXTRACT_GROUPED_TSV")"
    i=0
    echo ""
    echo "--- extraction: $total job(s) ---"
    while IFS=$'\t' read -r model dataset task pooling split; do
        [ -n "${model:-}" ] || continue
        ((i+=1))
        echo ""
        echo "[extract $i/$total] model=$model dataset=$dataset task=$task pooling=$pooling split=$split"
        run_extract_job "$model" "$dataset" "$task" "$pooling" "$split"
    done < "$EXTRACT_GROUPED_TSV"
fi

if [ "$SKIP_GEOMETRY" = "1" ]; then
    echo ""
    echo "--- geometry skipped: SKIP_GEOMETRY=1 ---"
else
    total="$(count_rows "$GEOMETRY_TSV")"
    i=0
    echo ""
    echo "--- geometry: $total job(s) ---"
    while IFS=$'\t' read -r model dataset task pooling; do
        [ -n "${model:-}" ] || continue
        ((i+=1))
        echo ""
        echo "[geometry $i/$total] model=$model dataset=$dataset task=$task pooling=$pooling geometry=$GEOMETRY_CONFIG"
        run_geometry_job "$model" "$dataset" "$task" "$pooling"
    done < "$GEOMETRY_TSV"
fi

if [ "$SKIP_PROBES" = "1" ]; then
    echo ""
    echo "--- probes skipped: SKIP_PROBES=1 ---"
else
    total="$(count_rows "$PROBE_TSV")"
    i=0
    echo ""
    echo "--- probes: $total job(s) ---"
    while IFS=$'\t' read -r model dataset task probe pooling seeds; do
        [ -n "${model:-}" ] || continue
        ((i+=1))
        echo ""
        echo "[probe $i/$total] model=$model dataset=$dataset task=$task probe=$probe pooling=$pooling seeds=$seeds"
        run_probe_job "$model" "$dataset" "$task" "$probe" "$pooling" "$seeds"
    done < "$PROBE_TSV"
fi

if [ "$PRUNE_EMBEDDINGS_AFTER_GROUP" = "1" ]; then
    echo ""
    echo "--- pruning staged intermediate embeddings ---"
    run_or_record "prune" "all embeddings" \
        python scripts/prune_embedding_cache.py \
            "--cache-root=${CACHE_ROOT}" \
            --all \
            --include-orphans \
            --yes
fi

run_export
}

run_streamed_pipeline() {
    echo ""
    echo "--- streamed execution by model/dataset combo ---"
    GROUP_TSV="$PLAN_DIR/model_dataset_jobs.tsv"
    {
        cut -f1,2 "$EXTRACT_TSV"
        cut -f1,2 "$GEOMETRY_TSV"
        cut -f1,2 "$PROBE_TSV"
    } | sort -u > "$GROUP_TSV"

    local total_groups
    local group_i
    total_groups="$(count_rows "$GROUP_TSV")"
    group_i=0
    while IFS=$'\t' read -r model dataset; do
        [ -n "${model:-}" ] || continue
        ((group_i+=1))
        local failures_before
        local failures_after
        failures_before="$(failure_count)"
        echo ""
        echo "=== combo $group_i/$total_groups: model=$model dataset=$dataset ==="

        if group_complete "$model" "$dataset"; then
            echo "[resume] downstream results complete; skipping extraction/geometry/probes for $model/$dataset"
            prune_group_embeddings "$model" "$dataset"
            clear_runtime_cache
            continue
        fi

        if [ "$SKIP_EXTRACT" = "1" ]; then
            echo "--- extraction skipped: SKIP_EXTRACT=1 ---"
        else
            while IFS=$'\t' read -r row_model row_dataset row_task row_pooling row_split; do
                [ -n "${row_model:-}" ] || continue
                echo "[extract] model=$row_model dataset=$row_dataset task=$row_task pooling=$row_pooling split=$row_split"
                run_extract_job "$row_model" "$row_dataset" "$row_task" "$row_pooling" "$row_split"
            done < <(awk -F '\t' -v m="$model" -v d="$dataset" '$1 == m && $2 == d' "$EXTRACT_GROUPED_TSV")
        fi

        if [ "$SKIP_GEOMETRY" = "1" ]; then
            echo "--- geometry skipped: SKIP_GEOMETRY=1 ---"
        else
            while IFS=$'\t' read -r row_model row_dataset row_task row_pooling; do
                [ -n "${row_model:-}" ] || continue
                echo "[geometry] model=$row_model dataset=$row_dataset task=$row_task pooling=$row_pooling"
                run_geometry_job "$row_model" "$row_dataset" "$row_task" "$row_pooling"
            done < <(awk -F '\t' -v m="$model" -v d="$dataset" '$1 == m && $2 == d' "$GEOMETRY_TSV")
        fi

        if [ "$SKIP_PROBES" = "1" ]; then
            echo "--- probes skipped: SKIP_PROBES=1 ---"
        else
            while IFS=$'\t' read -r row_model row_dataset row_task row_probe row_pooling row_seeds; do
                [ -n "${row_model:-}" ] || continue
                echo "[probe] model=$row_model dataset=$row_dataset task=$row_task probe=$row_probe pooling=$row_pooling seeds=$row_seeds"
                run_probe_job "$row_model" "$row_dataset" "$row_task" "$row_probe" "$row_pooling" "$row_seeds"
            done < <(awk -F '\t' -v m="$model" -v d="$dataset" '$1 == m && $2 == d' "$PROBE_TSV")
        fi

        failures_after="$(failure_count)"
        if [ "$failures_after" = "$failures_before" ] || [ "$PRUNE_ON_FAILURE" = "1" ]; then
            prune_group_embeddings "$model" "$dataset"
        else
            echo "[prune] skipped for $model/$dataset because this combo had failures"
        fi
        clear_runtime_cache
    done < "$GROUP_TSV"

    run_export
}

if [ "$STREAM_BY_COMBO" = "1" ]; then
    run_streamed_pipeline
else
    run_staged_pipeline
fi

echo ""
echo "=========="
if [ -s "$FAILURES_FILE" ]; then
    echo "SWEEP FINISHED WITH FAILURES"
    echo "Failures: $FAILURES_FILE"
    cat "$FAILURES_FILE"
    echo "Finished: $(date -Iseconds)"
    echo "=========="
    exit 1
fi

echo "SWEEP COMPLETE"
echo "Finished: $(date -Iseconds)"
echo "Plan: $PLAN_DIR"
echo "Log: $LOG_FILE"
echo "=========="
