"""Load exported results, build per-group oracles, and score layer selectors.

Terminology used throughout:

* **probe group** -- one (model, dataset, task, readout, representation side,
  probe type, split strategy) configuration. Its probe curve ``M(l)`` is the
  primary metric of the single-layer probe at layer ``l``, averaged over the
  three probe seeds.
* **geometry group** -- the same configuration without probe type / split
  strategy. Geometry is computed once per pooled representation, so a
  label-free selector chooses one layer per geometry group, and that layer is
  then evaluated against every probe group that shares the representation.

Evaluation metrics for a selected layer ``l_hat`` (Section 6 of the paper)::

    Gain-L  = M(l_hat) - M(L)            (reported x100, in percentage points)
    Gap-O   = max_j M(j) - M(l_hat)      (reported x100)
    Hit@1   = l_hat == oracle layer      (oracle ties -> shallowest layer)
    Hit@3   = l_hat in top-3 layers      (ties -> deeper layer ranks higher)
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import sgi as sgi_lib

GEOMETRY_KEYS = ["model_id", "dataset_id", "task_type", "pooling_name", "representation_side"]
PROBE_KEYS = [*GEOMETRY_KEYS, "probe_type", "split_strategy"]
NA = "__NA__"

#: Tasks analysed in the paper (clustering runs, if present, are ignored).
PAPER_TASKS = ("classification", "pair_classification", "sts", "multiple_choice_qa")

#: The 8 datasets used for the 383-group encoder SGI evaluation (HF ids).
ENCODER_SGI_DATASETS = (
    "stanfordnlp/sst2",
    "SetFit/ag_news",
    "dair-ai/emotion",
    "mteb/banking77",
    "stanfordnlp/snli",
    "nyu-mll/multi_nli",
    "SetFit/mrpc",
    "mteb/stsbenchmark-sts",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _fill_keys(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), NA).astype(str)
    return df


def load_probe_scores(results_dir: str | Path) -> pd.DataFrame:
    """Per-seed probe rows exported by ``pipeline/04_export_results.py``."""
    df = pd.read_parquet(Path(results_dir) / "probe_scores.parquet")
    df = df[df["task_type"].isin(PAPER_TASKS)]
    return _fill_keys(df, PROBE_KEYS)


def load_geometry(results_dir: str | Path) -> pd.DataFrame:
    """Wide per-layer geometry table exported by ``pipeline/04_export_results.py``."""
    df = pd.read_parquet(Path(results_dir) / "geometry.parquet")
    return _fill_keys(df, ["model_id", "dataset_id", "pooling_name", "representation_side", "split"])


def layer_scores(probe_scores: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged single-layer probe curves: one row per probe group and layer."""
    single = probe_scores[probe_scores["layer_aggregation"].eq("single_layer")]
    single = single[single["layer_index"].notna() & single["primary_metric"].notna()].copy()
    single["layer_index"] = single["layer_index"].astype(int)
    return (
        single.groupby([*PROBE_KEYS, "layer_index"], as_index=False)
        .agg(primary_metric=("primary_metric", "mean"), n_seeds=("seed", "nunique"))
        .sort_values([*PROBE_KEYS, "layer_index"], ignore_index=True)
    )


def learned_mix_scores(probe_scores: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged metric of the supervised all-layer scalar mix (ELMo-style)."""
    mix = probe_scores[probe_scores["layer_aggregation"].eq("learnable_weighting")]
    return mix.groupby(PROBE_KEYS, as_index=False).agg(learned_mix_metric=("primary_metric", "mean"))


def geometry_for_groups(geometry: pd.DataFrame, probe_groups: pd.DataFrame) -> pd.DataFrame:
    """Attach ``task_type`` to geometry rows and keep one embedding split per group.

    Geometry is computed on the training split when one exists (otherwise on the
    only available split, e.g. BIOSSES).
    """
    tasks = probe_groups[GEOMETRY_KEYS].drop_duplicates()
    geo = geometry.copy()
    geo["_split_rank"] = np.where(geo["split"].eq("train"), 0, 1)
    preferred = (
        geo.groupby(["model_id", "dataset_id", "pooling_name", "representation_side"])["_split_rank"]
        .transform("min")
    )
    geo = geo[geo["_split_rank"].eq(preferred)].drop(columns="_split_rank")
    merged = geo.merge(tasks, on=["model_id", "dataset_id", "pooling_name", "representation_side"])
    return merged.sort_values([*GEOMETRY_KEYS, "layer_index"], ignore_index=True)


# ---------------------------------------------------------------------------
# Oracle and evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Oracle:
    """Probe curve of one probe group plus its oracle / last / top-3 layers."""

    key: tuple
    scores: Mapping[int, float]
    best: int
    last: int
    top3: frozenset[int]


def build_oracles(scores: pd.DataFrame) -> dict[tuple, Oracle]:
    """Build an :class:`Oracle` for every probe group of :func:`layer_scores`."""
    oracles: dict[tuple, Oracle] = {}
    for key, group in scores.groupby(PROBE_KEYS, sort=False):
        curve = dict(zip(group["layer_index"].astype(int), group["primary_metric"].astype(float), strict=True))
        best_value = max(curve.values())
        best = min(layer for layer, value in curve.items() if value == best_value)
        ranked = sorted(curve.items(), key=lambda item: (item[1], item[0]), reverse=True)
        oracles[tuple(key)] = Oracle(
            key=tuple(key),
            scores=curve,
            best=best,
            last=max(curve),
            top3=frozenset(layer for layer, _ in ranked[:3]),
        )
    return oracles


def evaluate(
    method: str,
    oracles: Mapping[tuple, Oracle],
    choose: Callable[[Oracle], int | None],
) -> pd.DataFrame:
    """Score a selector on every oracle group. ``choose`` returns a layer or None."""
    rows = []
    for oracle in oracles.values():
        layer = choose(oracle)
        if layer is None or layer not in oracle.scores:
            continue
        s = oracle.scores
        rows.append(
            {
                **dict(zip(PROBE_KEYS, oracle.key, strict=True)),
                "method": method,
                "selected_layer": int(layer),
                "oracle_layer": oracle.best,
                "last_layer": oracle.last,
                "selected_metric": s[layer],
                "oracle_metric": s[oracle.best],
                "last_metric": s[oracle.last],
                "gain_last": s[layer] - s[oracle.last],
                "gap_oracle": s[oracle.best] - s[layer],
                "hit_at_1": layer == oracle.best,
                "hit_at_3": layer in oracle.top3,
            }
        )
    return pd.DataFrame(rows)


def selections_to_chooser(selected: pd.DataFrame) -> Callable[[Oracle], int | None]:
    """Turn a ``select_layers``-style frame into a chooser for :func:`evaluate`."""
    lookup = {
        tuple(row[col] for col in GEOMETRY_KEYS): int(row["selected_layer"])
        for _, row in selected.iterrows()
    }
    n_geo = len(GEOMETRY_KEYS)
    return lambda oracle: lookup.get(oracle.key[:n_geo])


def select_by_metric(geometry: pd.DataFrame, metric: str, *, higher: bool = True) -> pd.DataFrame:
    """Single-signal baseline: pick the latest layer that maximises (or minimises) one metric."""
    rows = []
    for keys, group in geometry.groupby(GEOMETRY_KEYS, sort=False):
        values = pd.to_numeric(group[metric], errors="coerce")
        if not values.notna().any():
            continue
        signed = values if higher else -values
        layer = sgi_lib.latest_argmax(pd.Series(signed.to_numpy(), index=group["layer_index"].astype(int)))
        rows.append({**dict(zip(GEOMETRY_KEYS, keys, strict=True)), "selected_layer": layer})
    return pd.DataFrame(rows)


def select_by_formula(
    geometry: pd.DataFrame,
    formula: Callable[[pd.DataFrame], pd.Series],
) -> pd.DataFrame:
    """Select layers with an arbitrary function of the SGI terms (see :func:`sgi.sgi_terms`)."""
    rows = []
    for keys, group in geometry.groupby(GEOMETRY_KEYS, sort=False):
        score = formula(sgi_lib.sgi_terms(group)).fillna(0.0)
        rows.append(
            {**dict(zip(GEOMETRY_KEYS, keys, strict=True)), "selected_layer": sgi_lib.latest_argmax(score)}
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries and significance tests
# ---------------------------------------------------------------------------


def summarize(frame: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Mean Gain-L / Gap-O (in percentage points) and Hit@1 / Hit@3."""
    by = by or ["method"]
    out = (
        frame.groupby(by, sort=False, dropna=False)
        .agg(
            n=("gain_last", "size"),
            gain_l_pp=("gain_last", "mean"),
            gap_o_pp=("gap_oracle", "mean"),
            hit_at_1=("hit_at_1", "mean"),
            hit_at_3=("hit_at_3", "mean"),
        )
        .reset_index()
    )
    out["gain_l_pp"] *= 100.0
    out["gap_o_pp"] *= 100.0
    return out


def paired_test(
    a: pd.Series | np.ndarray,
    b: pd.Series | np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired comparison ``a - b``: mean, bootstrap 95% CI, two-sided Wilcoxon p."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_value = float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    return {"n": int(len(diff)), "delta": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_value": p_value}


def json_scores(oracle: Oracle) -> str:
    """Serialise a probe curve (used when writing per-group selection files)."""
    return json.dumps({str(k): v for k, v in sorted(oracle.scores.items())})
