"""Abstract dataset wrapper and the normalized DatasetSample schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetSample:
    """One example normalized to a common schema for extraction and probing.

    ``inputs`` carries one or more text fields (single-text tasks use a list of
    length one; pair tasks use length two). ``label`` is task-dependent (int for
    classification, float for STS, list/dict for token classification and MCQ).
    ``extras`` is a free-form payload for things like option lists or language.
    """

    inputs: list[str]
    label: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class DatasetWrapper(ABC):
    """Abstract base for a registered dataset that can stream :class:`DatasetSample`s."""

    @abstractmethod
    def load_split(self, split: str) -> Iterable[DatasetSample]:
        """Yield samples from one split."""

    @abstractmethod
    def num_classes(self) -> int | None:
        """Number of label classes; ``None`` for regression or label-free tasks."""

    @abstractmethod
    def label_names(self) -> list[str] | None:
        """Human-readable class names if available."""
