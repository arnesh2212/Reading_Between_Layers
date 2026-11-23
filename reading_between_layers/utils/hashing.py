"""Stable content-addressable hashing of JSON-serializable objects."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..constants import HASH_LENGTH


def _canonicalize(obj: Any) -> Any:
    """Return a JSON-compatible representation with deterministic ordering."""
    if isinstance(obj, dict):
        return {str(k): _canonicalize(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def stable_hash(obj: Any, length: int = HASH_LENGTH) -> str:
    """Deterministic blake2b hash of a JSON-serializable object.

    Args:
        obj: any object that can be canonicalized to JSON. Dicts are sorted by key,
            lists preserve order, non-primitive leaves fall back to ``str``.
        length: number of hex characters of output. Must be even.

    Returns:
        Hex digest of the requested length.
    """
    if length <= 0 or length % 2 != 0:
        raise ValueError("length must be a positive even integer")
    canonical = _canonicalize(obj)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=length // 2).hexdigest()
    return digest
