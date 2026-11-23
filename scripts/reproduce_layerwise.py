"""Reproduce the layerwise peak analysis (Section 5, Figure 1-2, 6-7, Appendix C tables).

    python scripts/reproduce_layerwise.py --results results/paper --out outputs/paper

Writes CSV tables, a Markdown summary and figures under ``<out>/layerwise``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reading_between_layers import evaluation as ev
from reading_between_layers import layerwise as lw
from reading_between_layers import metadata as md
from reading_between_layers import (
    plotting,
    sgi,
)
from reading_between_layers.reporting import Report

#: The two qualitative overlay panels of Figure 2: (model, dataset, readout, probe).
FIGURE2_PANELS = [
    ("distilbert/distilbert-base-uncased", "mteb/banking77", "cls", "linear"),
    ("nghuyong/ernie-2.0-base-en", "stanfordnlp/snli", "cls", "linear"),
]


def coordinate_deltas(peaks: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    """Table 26: SGI coordinates at the empirical best layer minus at the final layer."""
    terms = []
    for keys, group in geometry.groupby(ev.GEOMETRY_KEYS, sort=False):
        t = sgi.sgi_terms(group)[list(sgi.COMPONENTS)]
        t["sgi"] = t.sum(axis=1)
        t = t.reset_index()
        for col, value in zip(ev.GEOMETRY_KEYS, keys, strict=True):
            t[col] = value
        terms.append(t)
    terms = pd.concat(terms, ignore_index=True).set_index(["model_id", "dataset_id", "task_type", "pooling_name", "layer_index"])
    rows = []
    for _, row in peaks.iterrows():
        base = (row["model_id"], row["dataset_id"], row["task_type"], row["pooling_name"])
        try:
            best = terms.loc[(*base, row["best_layer"])]
            last = terms.loc[(*base, row["last_layer"])]
        except KeyError:
            continue
        for name in [*sgi.COMPONENTS, "sgi"]:
            rows.append({"task_type": row["task_type"], "coordinate": name,
                         "delta": float(best[name] - last[name])})
    deltas = pd.DataFrame(rows)
    out = (
        deltas.groupby(["coordinate", "task_type"])["delta"]
        .agg(n="size", mean_delta="mean", median_delta="median", std="std")
        .reset_index()
    )
    out["task_type"] = out["task_type"].map(md.TASK_LABELS)
    return out


def overlay_curves(scores: pd.DataFrame, geometry: pd.DataFrame) -> list[dict]:
    curves = []
    for model_id, dataset_id, pooling, probe in FIGURE2_PANELS:
        s = scores[(scores.model_id == model_id) & (scores.dataset_id == dataset_id)
                   & (scores.pooling_name == pooling) & (scores.probe_type == probe)].sort_values("layer_index")
        g = geometry[(geometry.model_id == model_id) & (geometry.dataset_id == dataset_id)
                     & (geometry.pooling_name == pooling)]
        if s.empty or g.empty:
            continue
        terms = sgi.sgi_terms(g)
        score = terms[list(sgi.COMPONENTS)].sum(axis=1)
        layers = terms.index.to_numpy()
        n_layers = int(md.model_table().set_index("model_id").loc[model_id, "num_layers"])
        probe_curve = s.set_index("layer_index")["primary_metric"].reindex(layers).to_numpy()
        curves.append(
            {
                "title": f"Model: {md.short_model_name(model_id)}\nDataset: {md.DATASET_LABELS.get(dataset_id, dataset_id)}\n"
                         f"Task: {md.TASK_LABELS[s.task_type.iloc[0]]}; Pooling: {md.READOUT_LABELS[pooling]}",
                "depth": layers / n_layers,
                "probe": probe_curve,
                "sgi": score.to_numpy(),
                "terms": {name.capitalize(): terms[name].to_numpy() for name in sgi.COMPONENTS},
                "selected_depth": sgi.latest_argmax(score) / n_layers,
                "best_depth": layers[int(np.nanargmax(probe_curve))] / n_layers,
            }
        )
    return curves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results/paper"))
    parser.add_argument("--out", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    out = args.out / "layerwise"
    report = Report(out, title="Layerwise peak analysis: reproduced results")

    probe = ev.load_probe_scores(args.results)
    scores = ev.layer_scores(probe)
    geometry = ev.geometry_for_groups(ev.load_geometry(args.results), scores)
    peaks = lw.peak_instances(scores)
    out.mkdir(parents=True, exist_ok=True)
    peaks.to_csv(out / "peak_layer_instances.csv", index=False)

    scope = pd.DataFrame(
        [
            {"artifact": "Peak-layer instances", "rows": len(peaks), "models": peaks.model_id.nunique(),
             "datasets": peaks.dataset_id.nunique()},
            {"artifact": "Layerwise score rows", "rows": len(scores), "models": scores.model_id.nunique(),
             "datasets": scores.dataset_id.nunique()},
        ]
    )
    report.table("table12_scope", scope, "Table 12 - scope of the layerwise analysis.")
    report.table("table14_support", lw.support_table(peaks), "Table 14 - task x readout x probe support.")

    task_zones = lw.zone_table(peaks, "task_type", md.TASK_LABELS)
    readout_zones = lw.zone_table(peaks, "pooling_name", md.READOUT_LABELS)
    report.table("table16_task_zones", task_zones, "Table 16 / Figure 1 (top) - best-layer zone counts by task.")
    report.table("table17_readout_zones", readout_zones, "Table 17 / Figure 1 (bottom) - best-layer zone counts by readout.")
    report.table("table18_family_quartiles", lw.family_quartiles(peaks),
                 "Table 18 - normalised best-layer position by model family.")

    quads = lw.quadruplet_summary(peaks)
    quads.to_csv(out / "all_quadruplets.csv", index=False)
    supported = quads[quads["n"] >= 6]
    report.table("table19_peak_depth_patterns", supported.head(16).drop(columns="gap"),
                 "Table 19 - representative peak-depth patterns (strata with >= 6 instances).")
    report.table("table20_early_plateau_groups", supported.head(14).drop(columns=["median_peak", "early_peak_pct"]),
                 "Table 20 - highest early near-best plateau groups (>= 6 instances).")
    report.table("table21_early_near_best", lw.early_near_best_by_objective(peaks),
                 "Table 21 - share of instances whose earliest near-best layer lies in the first third.")
    report.table("table22_quadruplet_counts", lw.quadruplet_counts(quads),
                 "Table 22 - family/objective/task/readout strata at increasing minimum support.")
    report.table("table23_matched_readouts", lw.matched_readouts(peaks), "Table 23 - matched readout comparison.")
    pairwise = lw.pairwise_readouts(peaks)
    report.table("table24_pairwise_readouts", pairwise, "Table 24 / Figure 7 - pairwise readout comparisons.")

    encoder_peaks = peaks[peaks["family"].eq("encoder") & peaks["dataset_id"].isin(ev.ENCODER_SGI_DATASETS)]
    report.table("table26_coordinate_deltas", coordinate_deltas(encoder_peaks, geometry),
                 "Table 26 - SGI coordinate at the empirical best layer minus at the final layer (383 encoder groups).")
    report.write()

    if not args.no_figures:
        fig_dir = out / "figures"
        plotting.zone_bars(
            {"By task": task_zones.rename(columns={"task_type": "label"}),
             "By readout": readout_zones.rename(columns={"pooling_name": "label"})},
            fig_dir / "figure1_peak_zones",
        )
        plotting.overlay(overlay_curves(scores, geometry), fig_dir / "figure2_geometry_overlays")
        plotting.family_box(peaks, fig_dir / "figure6_family_best_layer")
        plotting.pairwise_readouts(pairwise, fig_dir / "figure7_pairwise_readouts")
    print(f"[layerwise] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
