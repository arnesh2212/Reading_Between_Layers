"""Abstract geometry metric interface and the GeometryResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from torch import Tensor, nn


@dataclass
class GeometryResult:
    """One metric's output along with provenance for downstream parquet rows."""

    metric_name: str
    values: dict[str, float]
    sample_size: int
    failure_reason: str | None = None
    extras: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class GeometryMetric(Protocol):
    """Protocol that every geometry metric implementation must satisfy.

    Implementations are stateless and may live in their own module file under
    ``reading_between_layers/geometry/``. ``compute`` returns a dictionary keyed by the entries of
    ``output_keys`` so the dispatcher in :func:`compute_all_geometry` can write
    one parquet row per output key.
    """

    name: str
    requires_labels: bool
    requires_jacobian: bool
    output_keys: list[str]

    def compute(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        """Return one float per output key.

        Args:
            embeddings: ``(n, d)`` pooled tensor or ``(n, seq, d)`` token tensor
                if the metric supports the token-level input (most do not).
            labels: integer or float labels when the metric uses them.
            attention_mask: optional mask aligned with ``embeddings`` when the
                metric expects token-level input.
            model: the model the embeddings were extracted from. Only the
                Jacobian-based metrics use this argument.

        Returns:
            Dictionary mapping each key in ``output_keys`` to a Python float.
            On failure the metric should log and return ``NaN`` for each key
            rather than raising; the caller stores the failure reason.
        """
        ...
