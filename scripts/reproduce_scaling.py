"""Summarise the larger-model scaling probe (Section 6.2, Table 7).

Run the scaling sweeps first (GPU), e.g.

    EXPERIMENT=scaling_encoders CACHE_ROOT=outputs/scaling_encoders/cache \\
        RESULTS_DIR=results/scaling_encoders scripts/run_sweep.sh
    EXPERIMENT=scaling_mamba CACHE_ROOT=outputs/scaling_mamba/cache \\
        RESULTS_DIR=results/scaling_mamba scripts/run_sweep.sh

then

    python scripts/reproduce_scaling.py \\
        --results results/scaling_encoders results/scaling_mamba --out outputs/paper

For every model, Gain-L is the unweighted mean over its datasets of the
per-dataset mean gain of SGI over the last layer. Encoder and SSM runs are
summarised per model and never pooled together.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reading_between_layers import evaluation as ev
from reading_between_layers import metadata as md
from reading_between_layers import sgi
from reading_between_layers.reporting import Report


def evaluate_results_dir(results_dir: Path) -> pd.DataFrame:
    scores = ev.layer_scores(ev.load_probe_scores(results_dir))
    oracles = ev.build_oracles(scores)
    geometry = ev.geometry_for_groups(ev.load_geometry(results_dir), scores)
    chooser = ev.selections_to_chooser(sgi.select_layers(geometry, ev.GEOMETRY_KEYS))
    frames = [
        ev.evaluate("SGI", oracles, chooser),
        ev.evaluate("Last", oracles, lambda o: o.last),
    ]
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, nargs="+",
                        default=[Path("results/scaling_encoders"), Path("results/scaling_mamba")])
    parser.add_argument("--out", type=Path, default=Path("outputs/paper"))
    args = parser.parse_args()

    available = [d for d in args.results if (d / "probe_scores.parquet").exists()]
    missing = [d for d in args.results if d not in available]
    for d in missing:
        print(f"[scaling] skipping {d}: no probe_scores.parquet (run the scaling sweep first)")
    if not available:
        return 1

    frame = pd.concat([evaluate_results_dir(d) for d in available], ignore_index=True)
    sgi_rows = frame[frame["method"].eq("SGI")]
    per_dataset = (
        sgi_rows.groupby(["model_id", "dataset_id"])
        .agg(n=("gain_last", "size"), gain_l_pp=("gain_last", "mean"), hit_at_1=("hit_at_1", "mean"),
             hit_at_3=("hit_at_3", "mean"))
        .reset_index()
    )
    per_dataset["gain_l_pp"] *= 100
    per_model = (
        per_dataset.groupby("model_id")
        .agg(datasets=("dataset_id", lambda s: ", ".join(md.DATASET_LABELS.get(d, d) for d in sorted(s))),
             paired_rows=("n", "sum"), gain_l_pp=("gain_l_pp", "mean"), hit_at_1=("hit_at_1", "mean"),
             hit_at_3=("hit_at_3", "mean"))
        .reset_index()
    )
    report = Report(args.out / "scaling", title="Larger-model scaling probe: reproduced results")
    report.table("table7_scaling", per_model, "Table 7 - mean SGI gain over the final layer per checkpoint.")
    report.table("scaling_by_dataset", per_dataset, "Per-model, per-dataset SGI results.")
    report.write()
    print(f"[scaling] wrote {args.out / 'scaling'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
