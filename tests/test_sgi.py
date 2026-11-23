"""Unit tests for the Stable Geometry Index and the selector evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reading_between_layers import evaluation as ev
from reading_between_layers import sgi


def _group(n_layers: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "layer_index": np.arange(1, n_layers + 1),
            "alpha_req_r2": rng.uniform(0.80, 0.99, n_layers),
            "effective_rank": rng.uniform(10, 300, n_layers),
            "condition_number": rng.uniform(1e2, 1e5, n_layers),
            "twonn_intrinsic_dim": rng.uniform(5, 40, n_layers),
        }
    )


def test_minmax_bounds_and_direction():
    x = pd.Series([1.0, 2.0, 3.0])
    hi = sgi.minmax(x, high=True)
    lo = sgi.minmax(x, high=False)
    assert hi.iloc[-1] == pytest.approx(1.0) and lo.iloc[0] == pytest.approx(1.0)
    assert hi.iloc[0] == pytest.approx(sgi.EPS)
    assert np.allclose(hi + lo, 1.0, atol=1e-9)


def test_constant_trace_maps_to_half():
    assert sgi.minmax(pd.Series([4.0, 4.0, 4.0])).eq(0.5).all()
    assert sgi.minmax(pd.Series([4.0, 4.0]), log=True).eq(0.5).all()


def test_rank_normalise_uses_average_ranks():
    r = sgi.rank_normalise(pd.Series([10.0, 10.0, 30.0]), high=True)
    assert r.tolist() == pytest.approx([1.0 / 3.0, 1.0 / 3.0, 2.5 / 3.0])


def test_log_minmax_is_monotone():
    x = pd.Series([1.0, 10.0, 100.0, 1000.0])
    out = sgi.minmax(x, log=True)
    assert out.is_monotonic_increasing and out.iloc[-1] == pytest.approx(1.0)


def test_depth_prior_peaks_at_four_fifths():
    terms = sgi.sgi_terms(_group(n_layers=10))
    assert int(terms["depth"].idxmax()) == 8


def test_score_is_sum_of_components():
    group = _group()
    terms = sgi.sgi_terms(group)
    score = sgi.sgi_score(group)
    assert np.allclose(score, terms[list(sgi.COMPONENTS)].sum(axis=1))


def test_min_max_and_rank_terms_are_scale_invariant():
    # kappa and x enter only through min-max / rank normalisers, so rescaling
    # them must not change the score (log-normalised traces are not scale-free).
    group = _group()
    rescaled = group.assign(condition_number=group["condition_number"] * 3.0,
                            alpha_req_r2=group["alpha_req_r2"] * 0.5)
    assert np.allclose(sgi.sgi_score(group), sgi.sgi_score(rescaled))


def test_latest_argmax_breaks_ties_towards_deeper_layer():
    scores = pd.Series([0.1, 0.9, 0.9, 0.2], index=[1, 2, 3, 4])
    assert sgi.latest_argmax(scores) == 3


def test_missing_metric_contributes_zero():
    group = _group()
    group.loc[2, "effective_rank"] = np.nan
    terms = sgi.sgi_terms(group)
    assert terms.loc[3, "capacity"] == 0.0


def test_select_layers_one_row_per_group():
    a = _group(seed=1).assign(model_id="m", dataset_id="d1")
    b = _group(seed=2).assign(model_id="m", dataset_id="d2")
    selected = sgi.select_layers(pd.concat([a, b]), ["model_id", "dataset_id"])
    assert len(selected) == 2 and selected["selected_layer"].between(1, 6).all()


def _oracle_frame(curve: dict[int, float]) -> pd.DataFrame:
    keys = dict(zip(ev.PROBE_KEYS, ["m", "d", "classification", "cls", ev.NA, "linear", "official_test"], strict=True))
    return pd.DataFrame([{**keys, "layer_index": k, "primary_metric": v} for k, v in curve.items()])


def test_oracle_tie_rules():
    oracle = next(iter(ev.build_oracles(_oracle_frame({1: 0.5, 2: 0.8, 3: 0.8, 4: 0.6})).values()))
    assert oracle.best == 2  # oracle ties -> shallowest layer
    assert oracle.last == 4
    assert oracle.top3 == {2, 3, 4}


def test_evaluate_metrics():
    oracles = ev.build_oracles(_oracle_frame({1: 0.5, 2: 0.8, 3: 0.7, 4: 0.6}))
    row = ev.evaluate("x", oracles, lambda o: 3).iloc[0]
    assert row["gain_last"] == pytest.approx(0.1)
    assert row["gap_oracle"] == pytest.approx(0.1)
    assert not row["hit_at_1"] and row["hit_at_3"]


def test_paired_test_detects_shift():
    rng = np.random.default_rng(0)
    b = rng.normal(size=200)
    res = ev.paired_test(b + 0.5, b, n_boot=500)
    assert res["delta"] == pytest.approx(0.5) and res["ci_low"] <= 0.5 <= res["ci_high"]
