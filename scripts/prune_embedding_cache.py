#!/usr/bin/env python
"""Remove intermediate embedding cache files without touching result tables."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from reading_between_layers.caching.keys import EmbeddingKey
from reading_between_layers.caching.shards import EMBEDDINGS_DIR, embedding_path
from reading_between_layers.datasets.registry import get_dataset_spec
from reading_between_layers.models.registry import get_model_spec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete cached embedding tensors and remove their manifest rows. "
            "Geometry/probe/selection/plot results are not touched."
        )
    )
    parser.add_argument("--cache-root", default="outputs/cache")
    parser.add_argument("--all", action="store_true", help="match all embedding rows")
    parser.add_argument("--model-name", help="registry model name to match")
    parser.add_argument("--dataset-name", help="registry dataset name to match")
    parser.add_argument("--pooling-name", help="pooling name to match")
    parser.add_argument("--split", help="dataset split to match")
    parser.add_argument(
        "--include-orphans",
        action="store_true",
        help="also delete embedding files that are not present in the manifest",
    )
    parser.add_argument("--yes", action="store_true", help="actually delete files")
    return parser.parse_args()


def _filters(args: argparse.Namespace) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if args.model_name:
        filters["model_id"] = get_model_spec(args.model_name).hf_id
    if args.dataset_name:
        spec = get_dataset_spec(args.dataset_name)
        filters["dataset_id"] = spec.hf_id
        filters["dataset_subset"] = spec.hf_subset
    if args.pooling_name:
        filters["pooling_name"] = args.pooling_name
    if args.split:
        filters["split"] = args.split
    if not args.all and not filters:
        raise SystemExit("Refusing to match everything without --all or a filter.")
    return filters


def _record_matches(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    if record.get("kind") != "embedding":
        return False
    payload = json.loads(record["key_json"])
    return all(payload.get(key) == value for key, value in filters.items())


def _remove_empty_dirs(path: Path, stop_at: Path) -> None:
    current = path
    stop_at = stop_at.resolve()
    while current.exists() and current.resolve() != stop_at:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _delete_manifest_matches(
    root: Path,
    filters: dict[str, Any],
    apply: bool,
) -> tuple[int, int, set[Path]]:
    manifest = root / "manifest.json"
    if not manifest.exists():
        return 0, 0, set()

    removed = 0
    bytes_removed = 0
    matched_paths: set[Path] = set()
    kept_lines: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not _record_matches(record, filters):
            kept_lines.append(line)
            continue
        payload = json.loads(record["key_json"])
        key = EmbeddingKey(**payload)
        path = embedding_path(root, key)
        if path.exists() and path not in matched_paths:
            matched_paths.add(path)
            bytes_removed += path.stat().st_size
            removed += 1
            if apply:
                path.unlink()
                _remove_empty_dirs(path.parent, root / EMBEDDINGS_DIR)

    if apply and removed:
        tmp = manifest.with_suffix(".json.tmp")
        tmp.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
        os.replace(tmp, manifest)
    return removed, bytes_removed, matched_paths


def _delete_orphans(root: Path, apply: bool, exclude: set[Path]) -> tuple[int, int]:
    embeddings_dir = root / EMBEDDINGS_DIR
    if not embeddings_dir.exists():
        return 0, 0
    files = [p for p in embeddings_dir.rglob("*") if p.is_file() and p not in exclude]
    bytes_removed = sum(p.stat().st_size for p in files)
    if apply:
        for path in files:
            path.unlink()
        for path in sorted(
            [p for p in embeddings_dir.rglob("*") if p.is_dir()],
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            try:
                path.rmdir()
            except OSError:
                pass
        try:
            embeddings_dir.rmdir()
        except OSError:
            pass
    return len(files), bytes_removed


def main() -> None:
    args = _parse_args()
    root = Path(args.cache_root)
    filters = _filters(args)
    manifest_count, manifest_bytes, manifest_paths = _delete_manifest_matches(
        root, filters, args.yes
    )
    orphan_count = orphan_bytes = 0
    if args.include_orphans and args.all:
        orphan_count, orphan_bytes = _delete_orphans(root, args.yes, manifest_paths)

    action = "deleted" if args.yes else "would delete"
    total_count = manifest_count + orphan_count
    total_bytes = manifest_bytes + orphan_bytes
    print(f"{action} {total_count} embedding file(s), {total_bytes / 1024**3:.2f} GiB from {root}")
    if not args.yes:
        print("Dry run only. Add --yes to apply.")


if __name__ == "__main__":
    main()
