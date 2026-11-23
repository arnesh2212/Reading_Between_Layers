"""Reproduce the Stable Geometry Index results (Section 6, Tables 1-9, Figures 3-5, 8).

    python scripts/reproduce_sgi.py --results results/paper --out outputs/paper

Reads only ``probe_scores.parquet`` and ``geometry.parquet`` from ``--results``
and writes CSV tables, a Markdown summary and figures under ``<out>/sgi``.
Runs on CPU in about a minute.
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
from reading_between_layers import metadata as md
from reading_between_layers import (
    plotting,
    sgi,
)
from reading_between_layers.reporting import Report

#: Single-signal label-free baselines of Table 1: name -> (geometry metric, higher is better).
BASELINES = {
    "Eff. rank": ("effective_rank", True),
    "RankMe": ("rankme", True),
    "Entropy": ("representation_entropy", True),
    "Part. ratio": ("participation_ratio", True),
    "Stable rank": ("stable_rank", True),
    "Low top-1 var ratio": ("top1_variance_ratio", False),
    "Low anis.": ("anisotropy_random_pairs", False),
    "Low cond.": ("condition_number", False),
    "Intr. dim.": ("twonn_intrinsic_dim", True),
    "IsoScore": ("isoscore", True),
    "Low iso-ratio": ("iso_eigenvalue_ratio", False),
    "Spectral floor": ("alpha_req_r2", False),
}

#: Ablations of Table 9: name -> (formula label, score as a function of the SGI terms).
ABLATIONS = {
    "SGI (full)": ("R + C + S + P", lambda t: t.residual + t.capacity + t.stability + t.depth),
    "No residual": ("C + S + P", lambda t: t.capacity + t.stability + t.depth),
    "No capacity": ("R + S + P", lambda t: t.residual + t.stability + t.depth),
    "No spread": ("R + C + P", lambda t: t.residual + t.capacity + t.depth),
    "No depth prior": ("R + C + S", lambda t: t.residual + t.capacity + t.stability),
    "Linear capacity": ("R + Z + S + P", lambda t: t.residual + t.z + t.stability + t.depth),
    "Product interaction": ("R + C + SP", lambda t: t.residual + t.capacity + t.stability * t.depth),
    "Residual x2": ("2R + C + S + P", lambda t: 2 * t.residual + t.capacity + t.stability + t.depth),
    "Capacity x2": ("R + 2C + S + P", lambda t: t.residual + 2 * t.capacity + t.stability + t.depth),
    "Stability x2": ("R + C + 2S + P", lambda t: t.residual + t.capacity + 2 * t.stability + t.depth),
    "Depth x2": ("R + C + S + 2P", lambda t: t.residual + t.capacity + t.stability + 2 * t.depth),
    # Single-coordinate selectors (are any of the terms sufficient alone?)
    "Residual only": ("R", lambda t: t.residual),
    "Capacity only": ("C", lambda t: t.capacity),
    "Stability only": ("S", lambda t: t.stability),
    "Depth only": ("P", lambda t: t.depth),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results/paper"))
    parser.add_argument("--out", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--n-boot", type=int, default=10_000, help="bootstrap resamples for Table 2")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    out = args.out / "sgi"
    out.mkdir(parents=True, exist_ok=True)
    report = Report(out, title="Stable Geometry Index: reproduced results")

    # ---- data ---------------------------------------------------------------
    probe = ev.load_probe_scores(args.results)
    scores = ev.layer_scores(probe)
    oracles = ev.build_oracles(scores)
    geometry = ev.geometry_for_groups(ev.load_geometry(args.results), scores)
    models = md.model_table().set_index("model_id")

    def is_encoder_group(key: tuple) -> bool:
        model_id, dataset_id = key[0], key[1]
        family = models["family"].get(model_id)
        return family == "encoder" and dataset_id in ev.ENCODER_SGI_DATASETS

    encoder = {k: o for k, o in oracles.items() if is_encoder_group(k)}

    # ---- selections ---------------------------------------------------------
    sgi_selected = sgi.select_layers(geometry, ev.GEOMETRY_KEYS)
    sgi_selected.to_csv(out / "sgi_selected_layers.csv", index=False)
    choose_sgi = ev.selections_to_chooser(sgi_selected)

    frames = {
        "SGI (Our Method)": ev.evaluate("SGI (Our Method)", encoder, choose_sgi),
        "Last": ev.evaluate("Last", encoder, lambda o: o.last),
        "Oracle": ev.evaluate("Oracle", encoder, lambda o: o.best),
    }
    for name, (metric, higher) in BASELINES.items():
        chooser = ev.selections_to_chooser(ev.select_by_metric(geometry, metric, higher=higher))
        frames[name] = ev.evaluate(name, encoder, chooser)
    per_group = pd.concat(frames.values(), ignore_index=True)
    per_group.to_csv(out / "encoder_selection_results.csv", index=False)

    sgi_enc = frames["SGI (Our Method)"]
    last_enc = frames["Last"]

    # ---- Table 1: encoder results --------------------------------------------
    table1 = ev.summarize(per_group).rename(columns={"method": "Method"})
    order = ["SGI (Our Method)"] + [m for m in table1.sort_values("gain_l_pp", ascending=False)["Method"]
                                    if m not in {"SGI (Our Method)", "Oracle"}] + ["Oracle"]
    table1 = table1.set_index("Method").loc[order].reset_index()
    report.table("table1_encoder_results", table1,
                 "Table 1 - encoder results (383 groups). Gain-L / Gap-O in percentage points.")

    # ---- Table 2: paired significance tests ---------------------------------
    paired = sgi_enc.merge(last_enc, on=ev.PROBE_KEYS, suffixes=("_sgi", "_last"))
    rows = []
    for metric, a, b in [
        ("Primary metric", paired["selected_metric_sgi"], paired["selected_metric_last"]),
        ("Hit@3", paired["hit_at_3_sgi"].astype(float), paired["hit_at_3_last"].astype(float)),
    ]:
        res = ev.paired_test(a, b, n_boot=args.n_boot)
        rows.append({"Metric": metric, "n": res["n"], "delta_pp": 100 * res["delta"],
                     "ci_low_pp": 100 * res["ci_low"], "ci_high_pp": 100 * res["ci_high"],
                     "wilcoxon_p": res["p_value"]})
    for name in BASELINES:
        other = frames[name]
        m = sgi_enc.merge(other, on=ev.PROBE_KEYS, suffixes=("_sgi", "_other"))
        res = ev.paired_test(m["selected_metric_sgi"], m["selected_metric_other"], n_boot=args.n_boot)
        rows.append({"Metric": f"Primary metric vs {name}", "n": res["n"], "delta_pp": 100 * res["delta"],
                     "ci_low_pp": 100 * res["ci_low"], "ci_high_pp": 100 * res["ci_high"],
                     "wilcoxon_p": res["p_value"]})
    report.table("table2_significance", pd.DataFrame(rows),
                 "Table 2 - paired tests of SGI vs. the last layer (and vs. each baseline); "
                 "paired bootstrap 95% CI and two-sided Wilcoxon signed-rank p.")

    # ---- Table 3: per-readout ------------------------------------------------
    t3 = ev.summarize(sgi_enc, by=["pooling_name"])
    t3["pooling_name"] = t3["pooling_name"].map(md.READOUT_LABELS)
    report.table("table3_pooling", t3, "Table 3 - SGI on encoders by readout.")

    # ---- Encoder breakdowns (task / dataset / subfamily) -----------------------
    enc_meta = md.attach_metadata(sgi_enc)
    report.table("encoder_by_task", ev.summarize(enc_meta, by=["task_type"]), "SGI on encoders by task.")
    by_dataset = ev.summarize(enc_meta, by=["dataset_id", "task_type"])
    by_dataset["dataset_id"] = by_dataset["dataset_id"].map(md.DATASET_LABELS).fillna(by_dataset["dataset_id"])
    report.table("encoder_by_dataset", by_dataset, "SGI on encoders by dataset.")
    subfam = ev.summarize(enc_meta, by=["encoder_subfamily"]).sort_values("gain_l_pp", ascending=False)
    report.table("figure4_encoder_subfamilies", subfam, "Figure 4 data - SGI by encoder subfamily.")

    # ---- Tables 4 and 6: all model families ----------------------------------
    sgi_all = md.attach_metadata(ev.evaluate("SGI", oracles, choose_sgi))
    sgi_all.to_csv(out / "all_groups_sgi_results.csv", index=False)
    table4 = ev.summarize(sgi_all, by=["task_type"])
    report.table("table4_tasks_all_families", table4,
                 f"Table 4 - SGI by task over all {len(sgi_all)} model/dataset/task/readout/probe groups.")
    fam = sgi_all.copy()
    is_main_encoder = fam.apply(lambda r: is_encoder_group(tuple(r[c] for c in ev.PROBE_KEYS)), axis=1)
    fam = fam[(fam["selector_family"] != "encoder") | is_main_encoder]
    table6 = ev.summarize(fam, by=["selector_family"])
    report.table("table6_model_families", table6,
                 "Table 6 - SGI by model family (encoder = the 383-group evaluation; contrastive "
                 "sentence encoders grouped as 'sentence_transformer').")

    # ---- Tables 5 and 8: supervised learned all-layer mix ---------------------
    mix = ev.learned_mix_scores(probe)
    cls = sgi_enc[sgi_enc["pooling_name"].eq("cls")].merge(mix, on=ev.PROBE_KEYS, how="inner")
    t5 = pd.DataFrame(
        [
            {"Method": "Oracle best layer", "metric": cls["oracle_metric"].mean()},
            {"Method": "SGI", "metric": cls["selected_metric"].mean()},
            {"Method": "Last layer", "metric": cls["last_metric"].mean()},
            {"Method": "Learned all-layer mix", "metric": cls["learned_mix_metric"].mean()},
        ]
    )
    t5["gain_l_pp"] = 100 * (t5["metric"] - cls["last_metric"].mean())
    report.table("table5_learned_mix", t5, f"Table 5 - SGI vs. supervised all-layer mix, matched CLS subset (n = {len(cls)}).")
    t8 = (
        cls.groupby("task_type")
        .agg(n=("selected_metric", "size"), SGI=("selected_metric", "mean"), Learned=("learned_mix_metric", "mean"))
        .reset_index()
    )
    report.table("table8_learned_mix_by_task", t8, "Table 8 - task-wise supervised comparison (matched CLS subset).")

    # ---- Table 9: ablations ----------------------------------------------------
    ablation_frames = {}
    for name, (_formula, fn) in ABLATIONS.items():
        chooser = ev.selections_to_chooser(ev.select_by_formula(geometry, fn))
        ablation_frames[name] = ev.evaluate(name, encoder, chooser)
    t9 = ev.summarize(pd.concat(ablation_frames.values(), ignore_index=True)).rename(columns={"method": "Variant"})
    t9.insert(1, "Score", t9["Variant"].map(lambda v: ABLATIONS[v][0]))
    report.table("table9_ablations", t9, "Table 9 - SGI ablations on the 383 encoder groups.")

    # ---- Figure 5 / 9: gain by selected depth -----------------------------------
    depth = sgi_enc.assign(depth=sgi_enc["selected_layer"] / sgi_enc["last_layer"], gain_pp=100 * sgi_enc["gain_last"])
    bins = np.linspace(0, 1, 6)
    labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"]
    depth["depth_bin"] = pd.cut(depth["depth"], bins=bins, labels=labels, include_lowest=True)
    depth_bins = depth.groupby("depth_bin", observed=False).agg(gain_pp=("gain_pp", "mean"), n=("gain_pp", "size")).reset_index()
    report.table("figure5_depth_bins", depth_bins, "Figure 5 data - mean SGI gain by normalised selected depth.")

    report.write()
    if not args.no_figures:
        fig_dir = out / "figures"
        plotting.depth_distribution(sgi_enc, fig_dir / "figure3_sgi_vs_oracle_depth")
        plotting.subfamily_gains(subfam, fig_dir / "figure4_encoder_subfamilies")
        plotting.depth_bin_gains(depth_bins, fig_dir / "figure5_depth_bin_gain")
        tradeoff = pd.concat(
            [
                table1[table1["Method"].isin(["SGI (Our Method)", "Last", "Eff. rank", "Entropy", "Part. ratio",
                                              "Low cond.", "IsoScore", "Spectral floor"])].assign(kind="baseline"),
                t9[t9["Variant"].isin(["No residual", "No capacity", "No spread", "No depth prior"])]
                .rename(columns={"Variant": "Method"}).assign(kind="ablation"),
            ],
            ignore_index=True,
        )
        tradeoff.loc[tradeoff["Method"].eq("SGI (Our Method)"), "kind"] = "sgi"
        plotting.gain_hit3_tradeoff(tradeoff, fig_dir / "figure8_gain_vs_hit3")
    print(f"[sgi] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
