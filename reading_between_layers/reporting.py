"""Write reproduced tables as CSV files plus one Markdown summary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        if value != 0 and abs(value) < 1e-3:
            return f"{value:.1e}"
        return f"{value:.3f}"
    return str(value)


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


class Report:
    """Accumulate named tables; ``write()`` emits ``<name>.csv`` and ``SUMMARY.md``."""

    def __init__(self, out_dir: Path, title: str) -> None:
        self.out_dir = Path(out_dir)
        self.title = title
        self.sections: list[tuple[str, str, pd.DataFrame]] = []

    def table(self, name: str, df: pd.DataFrame, caption: str) -> pd.DataFrame:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_dir / f"{name}.csv", index=False)
        self.sections.append((name, caption, df))
        return df

    def write(self) -> Path:
        parts = [f"# {self.title}", ""]
        for name, caption, df in self.sections:
            parts += [f"## {caption}", "", f"`{name}.csv`", "", to_markdown(df), ""]
        path = self.out_dir / "SUMMARY.md"
        path.write_text("\n".join(parts), encoding="utf-8")
        return path
