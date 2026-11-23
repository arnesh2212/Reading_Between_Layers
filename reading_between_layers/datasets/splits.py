"""Helpers for choosing runnable dataset splits from config and registry specs."""

from __future__ import annotations

from typing import Any

from .registry import DatasetSpec


def resolve_train_split(dataset_cfg: Any, dataset_spec: DatasetSpec) -> str:
    """Return the split used by extraction/probing stages.

    Most datasets expose a train split, but some evaluation-only datasets
    (for example BioSSES) only expose test. Prefer the configured train split,
    then a literal train split, then configured validation/test splits, then
    the first registry split.
    """
    configured_train = _get(dataset_cfg, "split_train")
    if configured_train:
        return str(configured_train)
    if "train" in dataset_spec.splits_available:
        return "train"
    for key in ("split_val", "split_test"):
        configured = _get(dataset_cfg, key)
        if configured:
            return str(configured)
    if dataset_spec.splits_available:
        return dataset_spec.splits_available[0]
    return "train"


def _get(config: Any, key: str) -> Any:
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key)
    return getattr(config, key, None)
