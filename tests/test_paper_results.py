"""Regression tests: the shipped results must reproduce the paper's headline numbers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reading_between_layers import evaluation as ev
from reading_between_layers import layerwise as lw
from reading_between_layers import metadata as md
from reading_between_layers import sgi

RESULTS = Path(__file__).resolve().parent.parent / "results" / "paper"
pytestmark = pytest.mark.skipif(not (RESULTS / "probe_scores.parquet").exists(), reason="results/paper not present")


@pytest.fixture(scope="module")
def data():
    scores = ev.layer_scores(ev.load_probe_scores(RESULTS))
    geometry = ev.geometry_for_groups(ev.load_geometry(RESULTS), scores)
    oracles = ev.build_oracles(scores)
    models = md.model_table().set_index("model_id")
    encoder = {
        k: o for k, o in oracles.items()
        if models["family"].get(k[0]) == "encoder" and k[1] in ev.ENCODER_SGI_DATASETS
    }
    return scores, geometry, encoder


def test_scope(data):
    scores, _, encoder = data
    assert len(scores) == 8544
    assert len(lw.peak_instances(scores)) == 617
    assert len(encoder) == 383


def test_table1_sgi_row(data):
    _, geometry, encoder = data
    chooser = ev.selections_to_chooser(sgi.select_layers(geometry, ev.GEOMETRY_KEYS))
    row = ev.summarize(ev.evaluate("SGI", encoder, chooser)).iloc[0]
    assert row["n"] == 383
    assert row["gain_l_pp"] == pytest.approx(1.1952, abs=1e-4)
    assert row["gap_o_pp"] == pytest.approx(1.8026, abs=1e-4)
    assert row["hit_at_1"] == pytest.approx(0.2846, abs=1e-4)
    assert row["hit_at_3"] == pytest.approx(0.5170, abs=1e-4)


def test_table1_last_and_effective_rank(data):
    _, geometry, encoder = data
    last = ev.summarize(ev.evaluate("Last", encoder, lambda o: o.last)).iloc[0]
    assert last["gap_o_pp"] == pytest.approx(2.9978, abs=1e-4)
    assert last["hit_at_3"] == pytest.approx(0.3890, abs=1e-4)
    chooser = ev.selections_to_chooser(ev.select_by_metric(geometry, "effective_rank"))
    er = ev.summarize(ev.evaluate("ER", encoder, chooser)).iloc[0]
    assert er["gain_l_pp"] == pytest.approx(0.4527, abs=1e-4)


def test_figure1_zones(data):
    scores, _, _ = data
    peaks = lw.peak_instances(scores)
    zones = lw.zone_table(peaks, "pooling_name", md.READOUT_LABELS).set_index("pooling_name")
    assert zones.loc["Mean pooled tokens", "early_pct"] == pytest.approx(61.1, abs=0.05)
    assert zones.loc["CLS token", "late_pct"] == pytest.approx(66.8, abs=0.05)


def test_learned_mix_subset(data):
    _, geometry, encoder = data
    chooser = ev.selections_to_chooser(sgi.select_layers(geometry, ev.GEOMETRY_KEYS))
    frame = ev.evaluate("SGI", encoder, chooser)
    mix = ev.learned_mix_scores(ev.load_probe_scores(RESULTS))
    cls = frame[frame["pooling_name"].eq("cls")].merge(mix, on=ev.PROBE_KEYS)
    assert len(cls) == 191
    assert cls["learned_mix_metric"].mean() == pytest.approx(0.683, abs=5e-4)
    assert isinstance(cls, pd.DataFrame)
