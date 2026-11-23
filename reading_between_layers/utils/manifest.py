"""Sweep manifest dataclass and IO helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .hashing import stable_hash

MANIFEST_FILENAME = "sweep_manifest.json"


@dataclass
class Manifest:
    """Resolved sweep description: tuples, estimates, warnings, and CLI commands."""

    run_id: str
    timestamp: str
    git_commit: str | None
    experiment: str
    tuples: list[dict[str, Any]] = field(default_factory=list)
    estimates: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)

    def hash(self) -> str:
        """Stable hash over the tuple list and experiment identity."""
        return stable_hash(
            {
                "tuples": self.tuples,
                "experiment": self.experiment,
                "run_id": self.run_id,
            }
        )


def write_manifest(manifest: Manifest, directory: Path) -> Path:
    """Write ``manifest`` to ``directory/sweep_manifest.json`` and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_FILENAME
    payload = asdict(manifest)
    payload["hash"] = manifest.hash()
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return path


def read_manifest(path: Path) -> Manifest:
    """Load a manifest previously written by :func:`write_manifest`."""
    data = json.loads(Path(path).read_text())
    data.pop("hash", None)
    return Manifest(**data)


def find_latest_manifest_hash(runs_dir: Path) -> str | None:
    """Return the hash of the most recent manifest under ``runs_dir``, or ``None``."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return None
    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        reverse=True,
    )
    for candidate in candidates:
        manifest_path = candidate / MANIFEST_FILENAME
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            return data.get("hash")
    return None
