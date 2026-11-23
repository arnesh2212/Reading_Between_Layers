"""Helpers for writing tabular result files with CSV sidecars."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_csv_sidecar(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` next to ``path`` using the same stem and a ``.csv`` suffix."""
    csv_path = path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path
