"""Stable Geometry Index (SGI): a closed-form, label-free layer selector.

For one evaluation group (a fixed model, dataset, task and readout) with
candidate layers ``1..L``, SGI scores every layer from four geometry traces
computed on unlabeled pooled representations:

========================  =========================================
trace                     meaning
========================  =========================================
``alpha_req_r2``          power-law fit quality of the spectrum (x)
``effective_rank``        exponentiated spectral entropy (e)
``condition_number``      sigma_1 / sigma_q (kappa)
``twonn_intrinsic_dim``   TwoNN local intrinsic dimension (delta)
========================  =========================================

Each trace is normalised *within the group* (across its layers) and combined
additively::

    R = N-(x)^2                              residual
    Z = L+(e),  C = Z (1 + Z)                capacity
    S = N-(kappa) + R-(kappa) + L+(delta)    stability
    P = N+(d^4 (1 - d)),  d = l / L          depth prior

    SGI = R + C + S + P,   selected layer = latest argmax of SGI

where ``N+`` is min-max scaling, ``N- = 1 - N+``, ``R-`` is one minus the
average-rank percentile and ``L+`` is min-max scaling after ``log(1 + u - min u)``.
No labels, probe scores or fitted coefficients are used.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

EPS = 1e-12

SGI_METRICS: tuple[str, ...] = (
    "alpha_req_r2",
    "effective_rank",
    "condition_number",
    "twonn_intrinsic_dim",
)
COMPONENTS: tuple[str, ...] = ("residual", "capacity", "stability", "depth")
DEFAULT_WEIGHTS: Mapping[str, float] = {name: 1.0 for name in COMPONENTS}


# ---------------------------------------------------------------------------
# Group-wise normalisers
# ---------------------------------------------------------------------------


def _finite(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna()


def minmax(values: pd.Series, *, high: bool = True, log: bool = False) -> pd.Series:
    """Min-max normaliser ``N+`` (``high=True``) or ``N-`` (``high=False``).

    With ``log=True`` this is the log-min-max normaliser ``L+``: values are first
    mapped through ``log(1 + u - min u)``. Constant traces map to 0.5 and
    non-finite entries stay NaN. Outputs are clipped to ``[EPS, 1]``.
    """
    out = pd.Series(np.nan, index=values.index, dtype=float)
    finite = _finite(values)
    if finite.empty:
        return out
    lo, hi = float(finite.min()), float(finite.max())
    if abs(hi - lo) <= EPS:
        scaled = pd.Series(0.5, index=finite.index, dtype=float)
    else:
        working = finite
        if log:
            working = np.log1p((working - lo).clip(lower=0.0))
            lo, hi = float(working.min()), float(working.max())
        if abs(hi - lo) <= EPS:
            scaled = pd.Series(0.5, index=finite.index, dtype=float)
        else:
            scaled = (working - lo) / (hi - lo)
    if not high:
        scaled = 1.0 - scaled
    out.loc[finite.index] = scaled.clip(lower=EPS, upper=1.0)
    return out


def rank_normalise(values: pd.Series, *, high: bool = True) -> pd.Series:
    """Rank normaliser ``R+`` / ``R-``: ``(rank - 1/2) / L`` with average ranks."""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    finite = _finite(values)
    if finite.empty:
        return out
    scaled = (finite.rank(method="average") - 0.5) / float(len(finite))
    if not high:
        scaled = 1.0 - scaled
    out.loc[finite.index] = scaled.clip(lower=EPS, upper=1.0)
    return out


# ---------------------------------------------------------------------------
# SGI coordinates
# ---------------------------------------------------------------------------


def sgi_terms(group: pd.DataFrame, layer_col: str = "layer_index") -> pd.DataFrame:
    """Return every SGI building block for one group, indexed by layer.

    ``group`` holds one row per layer with the four :data:`SGI_METRICS` columns.
    The returned frame contains the four SGI coordinates (``residual``,
    ``capacity``, ``stability``, ``depth``) plus the sub-terms used by the
    construction ablations (``z`` = linear capacity, ``kappa_value``,
    ``kappa_rank``, ``local_dim``, ``depth_mid``).
    """
    by_layer = group.assign(**{layer_col: group[layer_col].astype(int)}).set_index(layer_col)
    by_layer = by_layer.sort_index()
    layers = by_layer.index

    def col(name: str) -> pd.Series:
        if name in by_layer.columns:
            return by_layer[name]
        return pd.Series(np.nan, index=layers, dtype=float)

    x_low = minmax(col("alpha_req_r2"), high=False)
    z = minmax(col("effective_rank"), high=True, log=True)
    kappa_value = minmax(col("condition_number"), high=False)
    kappa_rank = rank_normalise(col("condition_number"), high=False)
    local_dim = minmax(col("twonn_intrinsic_dim"), high=True, log=True)

    depth = pd.Series(layers / float(max(1, int(layers.max()))), index=layers, dtype=float)
    depth = depth.clip(lower=EPS, upper=1.0)

    terms = pd.DataFrame(index=layers)
    terms["residual"] = x_low.pow(2.0).clip(lower=EPS, upper=1.0)
    terms["capacity"] = (z + z.pow(2.0)).clip(lower=EPS, upper=2.0)
    terms["stability"] = (kappa_value + kappa_rank + local_dim).clip(lower=EPS, upper=3.0)
    terms["depth"] = minmax(depth.pow(4.0) * (1.0 - depth), high=True)
    # Sub-terms used only by the construction ablations (Table 9).
    terms["z"] = z
    terms["kappa_value"] = kappa_value
    terms["kappa_rank"] = kappa_rank
    terms["local_dim"] = local_dim
    terms["depth_mid"] = minmax(depth.pow(2.0) * (1.0 - depth), high=True)
    terms.index.name = layer_col
    # A layer whose trace is missing contributes 0 for that term.
    return terms.fillna(0.0)


def sgi_score(
    group: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    layer_col: str = "layer_index",
) -> pd.Series:
    """SGI score per layer: ``sum_k w_k * component_k`` (all ``w_k = 1`` by default)."""
    weights = DEFAULT_WEIGHTS if weights is None else weights
    terms = sgi_terms(group, layer_col=layer_col)
    score = pd.Series(0.0, index=terms.index)
    for name, weight in weights.items():
        if weight:
            score = score + float(weight) * terms[name]
    return score.rename("sgi")


def latest_argmax(scores: pd.Series) -> int:
    """Index of the maximum score; ties go to the deepest layer."""
    frame = pd.DataFrame({"layer": scores.index.astype(int), "score": scores.to_numpy(float)})
    frame = frame.dropna(subset=["score"])
    if frame.empty:
        raise ValueError("cannot select a layer from an all-NaN score trace")
    return int(frame.sort_values(["score", "layer"], ascending=[False, False]).iloc[0]["layer"])


def select_layer(
    group: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    layer_col: str = "layer_index",
) -> int:
    """Select one layer for a single group with SGI."""
    return latest_argmax(sgi_score(group, weights=weights, layer_col=layer_col))


def select_layers(
    geometry: pd.DataFrame,
    group_cols: list[str],
    weights: Mapping[str, float] | None = None,
    layer_col: str = "layer_index",
) -> pd.DataFrame:
    """Run SGI for every group of a long per-layer geometry table.

    Returns one row per group with the group keys and ``selected_layer``.
    """
    rows = []
    for keys, group in geometry.groupby(group_cols, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(group_cols, keys, strict=True)),
                "selected_layer": select_layer(group, weights=weights, layer_col=layer_col),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "COMPONENTS",
    "DEFAULT_WEIGHTS",
    "SGI_METRICS",
    "latest_argmax",
    "minmax",
    "rank_normalise",
    "select_layer",
    "select_layers",
    "sgi_score",
    "sgi_terms",
]
