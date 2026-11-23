#!/usr/bin/env python
"""Best-effort cleanup between local sweep model/dataset groups."""

from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="outputs/cache")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass
    except Exception:
        pass

    cache_root = Path(args.cache_root)
    removed = 0
    for path in cache_root.glob("embedding_chunks_*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    print(f"[cleanup] runtime cache cleared; removed {removed} stale temp dir(s)")


if __name__ == "__main__":
    main()
