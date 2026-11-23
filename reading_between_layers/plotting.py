"""Matplotlib figures for the paper (one function per figure)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metadata import FAMILY_LABELS, READOUT_LABELS, TASK_LABELS  # noqa: E402

BLUE = "#2f5f8f"
GOLD = "#c49a2c"
RED = "#b85750"
SLATE = "#596a7a"
GREY = "#767676"
ZONE_COLORS = {"early": "#66b28f", "middle": "#6f9bd1", "late": "#e8a25a"}
TASK_COLORS = {"classification": BLUE, "pair_classification": GOLD, "sts": RED, "multiple_choice_qa": SLATE}

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e3e3e3",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 300,
    }
)


def save(fig: plt.Figure, stem: Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Layerwise study (Section 5, Appendix C)
# ---------------------------------------------------------------------------


def zone_bars(tables: dict[str, pd.DataFrame], stem: Path) -> None:
    """Figure 1: share of peak-layer instances in the early/middle/late third."""
    fig, axes = plt.subplots(len(tables), 1, figsize=(4.8, 1.25 * sum(len(t) for t in tables.values()) / 2 + 1.2))
    axes = np.atleast_1d(axes)
    for ax, (title, table) in zip(axes, tables.items(), strict=True):
        table = table.iloc[::-1]
        left = np.zeros(len(table))
        for zone in ("early", "middle", "late"):
            pct = table[f"{zone}_pct"].to_numpy()
            ax.barh(table.iloc[:, 0], pct, left=left, color=ZONE_COLORS[zone], label=zone.capitalize(), height=0.6)
            for y, (lft, p) in enumerate(zip(left, pct, strict=True)):
                if p >= 7:
                    ax.text(lft + p / 2, y, f"{p:.0f}%", ha="center", va="center", fontsize=6.5, color="white")
            left += pct
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of peak-layer instances (%)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.15), fontsize=7)
    fig.tight_layout()
    save(fig, stem)


def family_box(peaks: pd.DataFrame, stem: Path) -> None:
    """Figure 6: normalised best-layer position by model family."""
    order = ["encoder_decoder", "decoder", "ssm", "encoder", "sentence_transformer"]
    order = [f for f in order if f in set(peaks["family"])]
    data = [peaks.loc[peaks["family"].eq(f), "best_layer_frac"].dropna().to_numpy() for f in order]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.axhspan(0, 1 / 3, color=ZONE_COLORS["early"], alpha=0.08)
    ax.axhspan(1 / 3, 2 / 3, color=ZONE_COLORS["middle"], alpha=0.08)
    ax.axhspan(2 / 3, 1.0, color=ZONE_COLORS["late"], alpha=0.08)
    ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
               boxprops={"facecolor": "#cfe0ef", "edgecolor": BLUE}, medianprops={"color": BLUE})
    rng = np.random.default_rng(0)
    for i, (fam, values) in enumerate(zip(order, data, strict=True), start=1):
        sub = peaks[peaks["family"].eq(fam)]
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        colors = [TASK_COLORS.get(t, GREY) for t in sub["task_type"]]
        ax.scatter(np.full(len(values), i) + jitter, values, s=5, alpha=0.45, c=colors, linewidths=0)
    ax.set_xticks(range(1, len(order) + 1), [FAMILY_LABELS.get(f, f) for f in order], rotation=20)
    ax.set_ylabel("Normalized empirical best-layer depth")
    ax.set_ylim(0, 1.03)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=TASK_LABELS[t]) for t, c in TASK_COLORS.items()]
    ax.legend(handles=handles, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, stem)


def pairwise_readouts(pairwise: pd.DataFrame, stem: Path) -> None:
    """Figure 7: pairwise readout advantages within matched comparison groups."""
    df = pairwise.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    colors = [TASK_COLORS.get(t, GREY) for t in df["task_type"]]
    labels = [f"{TASK_LABELS.get(t, t)}\n{p}" for t, p in zip(df["task_type"], df["comparison"], strict=True)]
    ax.barh(labels, df["mean_delta"], color=colors, alpha=0.8, height=0.6)
    for y, (d, w, n) in enumerate(zip(df["mean_delta"], df["win_pct"], df["n"], strict=True)):
        ax.text(d + 0.004, y, f"Delta={d:.3f}; win={w:.1f}%, n={n}", va="center", fontsize=6.5)
    ax.set_xlabel("Mean best-layer metric advantage")
    ax.set_xlim(0, max(0.35, float(df["mean_delta"].max()) * 1.6))
    ax.tick_params(axis="y", labelsize=6.5)
    save(fig, stem)


def overlay(curves: list[dict], stem: Path) -> None:
    """Figure 2 / 10: probe curve and SGI (min-max normalised) against depth.

    Each entry of ``curves`` has keys ``title``, ``depth``, ``probe``, ``sgi``,
    ``terms`` (dict name -> array), ``selected_depth`` and ``best_depth``.
    """
    fig, axes = plt.subplots(1, len(curves), figsize=(3.4 * len(curves), 2.6), squeeze=False)
    for ax, c in zip(axes[0], curves, strict=True):
        for name, values in c["terms"].items():
            ax.plot(c["depth"], _unit(values), lw=0.8, alpha=0.45, label=name)
        ax.plot(c["depth"], _unit(c["probe"]), color="black", marker="o", ms=3, lw=1.3, label="Probe metric")
        ax.plot(c["depth"], _unit(c["sgi"]), color=BLUE, marker="s", ms=3, lw=1.3, label="Stable geometry index")
        ax.axvline(c["selected_depth"], color=BLUE, ls="--", lw=0.9)
        ax.axvline(c["best_depth"], color="#4f7f52", ls=":", lw=0.9)
        ax.set_title(c["title"], fontsize=6.8)
        ax.set_xlabel("Normalized depth")
        ax.set_xlim(0, 1.02)
    axes[0][0].set_ylabel("Min-max normalized value")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=6.5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, stem)


def _unit(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    return np.full_like(arr, 0.5) if hi - lo <= 1e-12 else (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# SGI selector (Section 6)
# ---------------------------------------------------------------------------


def depth_distribution(sgi_frame: pd.DataFrame, stem: Path) -> None:
    """Figure 3: depth histograms of SGI-selected (solid) vs oracle (dotted) layers."""
    df = sgi_frame.assign(
        sgi_depth=sgi_frame["selected_layer"] / sgi_frame["last_layer"],
        oracle_depth=sgi_frame["oracle_layer"] / sgi_frame["last_layer"],
    )
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for task, group in df.groupby("task_type"):
        color = TASK_COLORS.get(task, GREY)
        ax.hist(group["sgi_depth"], bins=bins, histtype="step", lw=1.6, color=color, label=f"{TASK_LABELS.get(task, task)} (SGI)")
        ax.hist(group["oracle_depth"], bins=bins, histtype="step", lw=1.2, ls=":", color=color, label=f"{TASK_LABELS.get(task, task)} (oracle)")
    ax.set_xlabel(r"Layer depth $\ell/L$")
    ax.set_ylabel("Encoder groups")
    ax.set_title("SGI and Oracle Depth Distributions")
    ax.legend(fontsize=6.3)
    save(fig, stem)


def subfamily_gains(subfam: pd.DataFrame, stem: Path) -> None:
    """Figure 4: mean SGI gain over the last layer per encoder subfamily."""
    df = subfam.sort_values("gain_l_pp")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    y = np.arange(len(df))
    ax.barh(y, df["gain_l_pp"], color="#9ab9d6", height=0.58)
    ax.scatter(df["gain_l_pp"], y, s=28 + 110 * df["hit_at_3"], color=BLUE, edgecolor="white", zorder=3)
    xmax = float(df["gain_l_pp"].max())
    for yi, (g, h) in enumerate(zip(df["gain_l_pp"], df["hit_at_3"], strict=True)):
        ax.text(xmax + 0.35, yi, f"{g:.2f}", va="center", ha="right", fontsize=6.8)
        ax.text(xmax + 0.9, yi, f"{h:.2f}", va="center", ha="right", fontsize=6.8)
    ax.text(xmax + 0.35, len(df) - 0.3, "gain", ha="right", fontsize=6.8, color="#555555")
    ax.text(xmax + 0.9, len(df) - 0.3, "H@3", ha="right", fontsize=6.8, color="#555555")
    ax.set_yticks(y, df["encoder_subfamily"])
    ax.set_xlabel("Mean gain over final layer (pp)")
    ax.set_title("Encoder Subfamilies")
    ax.set_xlim(min(0, float(df["gain_l_pp"].min()) - 0.2), xmax + 1.05)
    save(fig, stem)


def depth_bin_gains(bins: pd.DataFrame, stem: Path) -> None:
    """Figure 5 / 9: mean SGI gain binned by normalised selected depth."""
    df = bins.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(4.7, 2.9))
    top = df["gain_pp"].max()
    colors = [BLUE if v == top else "#d7e3eb" for v in df["gain_pp"]]
    ax.barh(np.arange(len(df)), df["gain_pp"], color=colors, edgecolor="#9ab3c7", height=0.58)
    ax.axvline(0, color="#303030", lw=0.8)
    for y, (g, n) in enumerate(zip(df["gain_pp"], df["n"], strict=True)):
        ax.text(g + 0.07 if g >= 0 else g / 2, y, f"{g:.2f} pp   n={int(n)}", va="center",
                ha="left" if g >= 0 else "center", fontsize=6.8)
    ax.set_yticks(np.arange(len(df)), df["depth_bin"].astype(str))
    ax.set_xlabel("Mean gain over final layer (pp)")
    ax.set_ylabel(r"SGI-selected depth bin $\hat{\ell}/L$")
    ax.set_title("Where Selected Depth Pays Off")
    ax.set_xlim(min(-1.05, float(df["gain_pp"].min()) - 0.45), max(2.75, float(top) + 0.65))
    save(fig, stem)


def gain_hit3_tradeoff(df: pd.DataFrame, stem: Path) -> None:
    """Figure 8: mean gain vs Hit@3 for SGI, single-signal baselines and ablations."""
    style = {"sgi": ("o", BLUE, 70), "baseline": ("o", SLATE, 34), "ablation": ("^", GOLD, 40)}
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for kind, group in df.groupby("kind"):
        marker, color, size = style[kind]
        ax.scatter(group["gain_l_pp"], group["hit_at_3"], marker=marker, s=size, color=color, edgecolor="white", zorder=3)
        for _, row in group.iterrows():
            label = "SGI" if row["Method"].startswith("SGI") else row["Method"]
            left = label in {"No residual", "SGI"}  # keep neighbouring labels apart
            ax.text(row["gain_l_pp"] + (-0.05 if left else 0.05), row["hit_at_3"] + 0.004, label,
                    fontsize=6.3, ha="right" if left else "left")
    ax.axvline(0, color="#303030", lw=0.8)
    ax.set_xlabel("Mean gain over final layer (pp)")
    ax.set_ylabel("Hit@3")
    ax.set_title("Gain and Hit@3 Move Together")
    save(fig, stem)


__all__ = [
    "depth_bin_gains",
    "depth_distribution",
    "family_box",
    "gain_hit3_tradeoff",
    "overlay",
    "pairwise_readouts",
    "save",
    "subfamily_gains",
    "zone_bars",
]
_ = READOUT_LABELS  # re-exported label map used by callers
