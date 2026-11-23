"""Export the stage-02/03 cache into two small, analysis-ready tables.

    python pipeline/04_export_results.py --cache-root outputs/cache --out results/paper

Writes

* ``<out>/probe_scores.parquet`` -- one row per (model, dataset, task, readout,
  probe, split strategy, layer, seed). ``layer_aggregation`` is
  ``single_layer`` for ordinary per-layer probes and ``learnable_weighting``
  for the supervised all-layer scalar mix.
* ``<out>/geometry.parquet`` -- one row per (model, dataset, split, readout,
  layer) with one column per geometry metric.

Everything in ``scripts/reproduce_*.py`` reads only these two files, so the
paper's tables and figures can be regenerated without a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reading_between_layers.caching.parquet_store import read_all
from reading_between_layers.caching.shards import geometry_dir, probes_dir

PROBE_COLUMNS = [
    "model_id",
    "dataset_id",
    "task_type",
    "pooling_name",
    "representation_side",
    "probe_type",
    "split_strategy",
    "layer_aggregation",
    "layer_index",
    "num_layers",
    "seed",
    "primary_metric",
    "val_primary",
    "ci_lower",
    "ci_upper",
    "learned_layer_weights_json",
]
GEOMETRY_INDEX = ["model_id", "dataset_id", "split", "pooling_name", "representation_side", "layer_index"]


def export_probes(cache_root: Path) -> pd.DataFrame:
    raw = read_all(probes_dir(cache_root))
    if raw.empty:
        raise SystemExit(f"no probe rows under {probes_dir(cache_root)}")
    records = []
    for payload in raw["value_json"].dropna():
        try:
            record = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    df = pd.DataFrame(records)
    for col in PROBE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[PROBE_COLUMNS]
    df["layer_index"] = pd.to_numeric(df["layer_index"], errors="coerce").astype("Int64")
    df["num_layers"] = pd.to_numeric(df["num_layers"], errors="coerce").astype("Int64")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    key = [c for c in PROBE_COLUMNS if c not in {"primary_metric", "val_primary", "ci_lower", "ci_upper",
                                                  "learned_layer_weights_json", "num_layers"}]
    df = df.drop_duplicates(subset=key, keep="last")
    return df.sort_values(key, ignore_index=True)


def export_geometry(cache_root: Path) -> pd.DataFrame:
    path = geometry_dir(cache_root) / "geometry.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}; run pipeline/02_compute_geometry.py first")
    long = pd.read_parquet(path)
    if "failure_reason" in long.columns:
        long = long[long["failure_reason"].isna()]
    long = long.copy()
    long["representation_side"] = long["representation_side"].astype("object")
    wide = long.pivot_table(
        index=GEOMETRY_INDEX,
        columns="metric_name",
        values="metric_value",
        aggfunc="last",
        dropna=False,
    ).reset_index()
    wide.columns.name = None
    metric_cols = [c for c in wide.columns if c not in GEOMETRY_INDEX]
    wide = wide.dropna(subset=metric_cols, how="all")
    sizes = long.groupby(GEOMETRY_INDEX, dropna=False)["sample_size"].max().rename("sample_size").reset_index()
    wide = wide.merge(sizes, on=GEOMETRY_INDEX, how="left")
    wide["representation_side"] = wide["representation_side"].replace("nan", pd.NA)
    return wide.sort_values(GEOMETRY_INDEX, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/cache"))
    parser.add_argument("--out", type=Path, default=Path("results/paper"))
    parser.add_argument("--drop-tasks", default="clustering",
                        help="comma-separated task types to leave out (not analysed in the paper)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    probes = export_probes(args.cache_root)
    drop = {t.strip() for t in args.drop_tasks.split(",") if t.strip()}
    probes = probes[~probes["task_type"].isin(drop)].reset_index(drop=True)
    probes.to_parquet(args.out / "probe_scores.parquet", compression="zstd", index=False)
    geometry = export_geometry(args.cache_root)
    # Keep geometry only for representations that have probe results.
    used = probes[["model_id", "dataset_id", "pooling_name"]].drop_duplicates()
    geometry = geometry.merge(used, on=["model_id", "dataset_id", "pooling_name"], how="inner")
    geometry.to_parquet(args.out / "geometry.parquet", compression="zstd", index=False)

    print(f"[export] probe rows    : {len(probes):>7}  -> {args.out / 'probe_scores.parquet'}")
    print(f"[export] geometry rows : {len(geometry):>7}  -> {args.out / 'geometry.parquet'}")
    print(f"[export] models        : {probes['model_id'].nunique()}")
    print(f"[export] datasets      : {probes['dataset_id'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
