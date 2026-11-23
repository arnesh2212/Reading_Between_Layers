"""Where does representation quality live? Peak-layer analysis (Section 5, Appendix C).

A *peak-layer instance* is one seed-averaged probe curve for a fixed
(model, dataset, task, readout, probe type). For each instance we record

* the empirical best layer (first maximiser) and its normalised depth
  ``best_layer / num_layers``;
* the zone of that depth: early (<= 1/3), middle (<= 2/3) or late;
* the earliest *near-best* layer, i.e. the first layer within 0.01 of the
  best score, and whether it lies in the first third (``early_1pp_plateau``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metadata as md

PEAK_KEYS = ["model_id", "dataset_id", "task_type", "pooling_name", "probe_type"]
ZONES = ("early", "middle", "late")
NEAR_BEST_TOLERANCE = 0.01


def zone(frac: float) -> str:
    if pd.isna(frac):
        return "unknown"
    if frac <= 1 / 3:
        return "early"
    if frac <= 2 / 3:
        return "middle"
    return "late"


def peak_instances(layer_scores: pd.DataFrame) -> pd.DataFrame:
    """One row per peak-layer instance (617 in the paper)."""
    scores = md.attach_metadata(layer_scores)
    rows = []
    for keys, sub in scores.groupby(PEAK_KEYS, sort=True):
        sub = sub.sort_values("layer_index")
        metric = sub["primary_metric"].to_numpy(dtype=float)
        layers = sub["layer_index"].to_numpy(dtype=int)
        best_i = int(np.argmax(metric))
        near_i = int(np.flatnonzero(metric >= metric[best_i] - NEAR_BEST_TOLERANCE)[0])
        n_layers = sub["num_layers"].iloc[0]
        n_layers = int(layers.max()) if pd.isna(n_layers) or n_layers <= 0 else int(n_layers)
        best_frac = layers[best_i] / n_layers
        near_frac = layers[near_i] / n_layers
        rows.append(
            {
                **dict(zip(PEAK_KEYS, keys, strict=True)),
                "family": sub["family"].iloc[0],
                "objective_code": sub["objective_code"].iloc[0],
                "pretraining_objective": sub["pretraining_objective"].iloc[0],
                "num_layers": n_layers,
                "best_layer": int(layers[best_i]),
                "best_metric": float(metric[best_i]),
                "last_layer": int(layers[-1]),
                "last_metric": float(metric[-1]),
                "gap_best_minus_last": float(metric[best_i] - metric[-1]),
                "earliest_near_best_layer": int(layers[near_i]),
                "best_layer_frac": best_frac,
                "earliest_near_best_frac": near_frac,
                "best_zone": zone(best_frac),
                "early_best_peak": bool(best_frac <= 1 / 3),
                "early_1pp_plateau": bool(near_frac <= 1 / 3),
            }
        )
    return pd.DataFrame(rows)


def zone_table(peaks: pd.DataFrame, by: str, labels: dict[str, str]) -> pd.DataFrame:
    """Tables 16/17 and Figure 1: early/middle/late counts and shares."""
    counts = peaks.groupby([by, "best_zone"]).size().unstack(fill_value=0).reindex(columns=list(ZONES), fill_value=0)
    counts["total"] = counts.sum(axis=1)
    out = counts.reset_index()
    out[by] = out[by].map(labels).fillna(out[by])
    for z in ZONES:
        out[f"{z}_pct"] = 100.0 * out[z] / out["total"]
    return out[[by, *ZONES, "total", *(f"{z}_pct" for z in ZONES)]]


def support_table(peaks: pd.DataFrame) -> pd.DataFrame:
    """Table 14: task x readout x probe support."""
    t = peaks.pivot_table(index=["task_type", "pooling_name"], columns="probe_type", values="best_layer",
                          aggfunc="size", fill_value=0).reset_index()
    t.columns.name = None
    t["total"] = t[[c for c in ("linear", "nonlinear") if c in t]].sum(axis=1)
    t["task_type"] = t["task_type"].map(md.TASK_LABELS)
    t["pooling_name"] = t["pooling_name"].map(md.READOUT_LABELS)
    return t


def family_quartiles(peaks: pd.DataFrame) -> pd.DataFrame:
    """Table 18: quartiles of normalised best-layer depth per family (Tukey outliers)."""
    rows = []
    for family, g in peaks.groupby("family"):
        v = g["best_layer_frac"].dropna()
        q1, q2, q3 = v.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        # Tukey fences; values lying on a fence (up to float error) are not outliers.
        tol = 1e-9
        outliers = int(((v < q1 - 1.5 * iqr - tol) | (v > q3 + 1.5 * iqr + tol)).sum())
        rows.append({"family": md.FAMILY_LABELS.get(family, family), "n": len(v), "q1": q1, "median": q2,
                     "q3": q3, "iqr": iqr, "outliers": outliers})
    return pd.DataFrame(rows).sort_values("median", ignore_index=True)


def quadruplet_summary(peaks: pd.DataFrame) -> pd.DataFrame:
    """Family x objective x task x readout strata (basis of Tables 19, 20, 22)."""
    # Group on registry codes so ties keep a stable, label-independent order.
    cols = ["family", "objective_code", "task_type", "pooling_name"]
    q = (
        peaks.groupby(cols)
        .agg(
            n=("best_layer", "size"),
            median_peak=("best_layer_frac", "median"),
            early_peak_pct=("early_best_peak", "mean"),
            early_1pp_pct=("early_1pp_plateau", "mean"),
            gap=("gap_best_minus_last", "mean"),
        )
        .reset_index()
    )
    q["early_peak_pct"] *= 100
    q["early_1pp_pct"] *= 100
    q = q.sort_values(["early_1pp_pct", "early_peak_pct", "n"], ascending=[False, False, False], ignore_index=True)
    q.insert(1, "pretraining_objective", q.pop("objective_code").map(md.OBJECTIVE_LABELS))
    q["family"] = q["family"].map(md.FAMILY_LABELS).fillna(q["family"])
    q["task_type"] = q["task_type"].map(md.TASK_LABELS)
    q["pooling_name"] = q["pooling_name"].map(md.READOUT_LABELS)
    return q


def quadruplet_counts(quads: pd.DataFrame, thresholds=(6, 8, 10)) -> pd.DataFrame:
    """Table 22: number of strata retained at increasing minimum support."""
    rows = [{"rule": "All observed quadruplets", "count": len(quads)}]
    rows += [{"rule": f"Quadruplets with at least {k} instances", "count": int((quads["n"] >= k).sum())}
             for k in thresholds]
    return pd.DataFrame(rows)


def early_near_best_by_objective(peaks: pd.DataFrame) -> pd.DataFrame:
    """Table 21: share of instances whose earliest near-best layer is in the first third."""
    t = (
        peaks.groupby(["task_type", "objective_code"])
        .agg(n=("early_1pp_plateau", "size"), early_near_best_pct=("early_1pp_plateau", "mean"))
        .reset_index()
    )
    t["early_near_best_pct"] *= 100
    t = t.sort_values(["task_type", "early_near_best_pct"], ascending=[True, False], kind="stable", ignore_index=True)
    t.insert(1, "pretraining_objective", t.pop("objective_code").map(md.OBJECTIVE_LABELS))
    t["task_type"] = t["task_type"].map(md.TASK_LABELS)
    return t


MATCH_KEYS = ["model_id", "dataset_id", "task_type", "probe_type"]


def matched_readouts(peaks: pd.DataFrame) -> pd.DataFrame:
    """Table 23: readout win rates within groups sharing model, dataset, task and probe."""
    p = peaks.copy()
    p["n_readouts"] = p.groupby(MATCH_KEYS)["pooling_name"].transform("nunique")
    p = p[p["n_readouts"] >= 2].copy()
    p["group_best"] = p.groupby(MATCH_KEYS)["best_metric"].transform("max")
    p["win"] = p["best_metric"].eq(p["group_best"])
    p["gap"] = p["best_metric"] - p["group_best"]
    t = (
        p.groupby(["task_type", "pooling_name"])
        .agg(n=("win", "size"), win_pct=("win", "mean"), mean_gap=("gap", "mean"),
             early_1pp_pct=("early_1pp_plateau", "mean"))
        .reset_index()
    )
    t["win_pct"] *= 100
    t["early_1pp_pct"] *= 100
    t = t.sort_values(["task_type", "win_pct"], ascending=[True, False], ignore_index=True)
    t["task_type"] = t["task_type"].map(md.TASK_LABELS)
    t["pooling_name"] = t["pooling_name"].map(md.READOUT_LABELS)
    return t


def pairwise_readouts(peaks: pd.DataFrame) -> pd.DataFrame:
    """Table 24 / Figure 7: pairwise readout comparisons, oriented to the better readout."""
    wide = peaks.pivot_table(index=MATCH_KEYS, columns="pooling_name", values="best_metric").reset_index()
    readouts = [c for c in ("cls", "last_token", "mean", "sbert_pair") if c in wide.columns]
    rows = []
    for task, g in wide.groupby("task_type"):
        for i, a in enumerate(readouts):
            for b in readouts[i + 1:]:
                both = g[[a, b]].dropna()
                if both.empty:
                    continue
                delta = both[a] - both[b]
                if delta.mean() < 0:
                    a_, b_, delta = b, a, -delta
                else:
                    a_, b_ = a, b
                rows.append(
                    {
                        "task_type": task,
                        "comparison": f"{md.READOUT_LABELS[a_]} vs {md.READOUT_LABELS[b_]}",
                        "n": len(delta),
                        "win_pct": 100 * float((delta > 0).mean()),
                        "mean_delta": float(delta.mean()),
                        "median_delta": float(delta.median()),
                    }
                )
    return pd.DataFrame(rows)
