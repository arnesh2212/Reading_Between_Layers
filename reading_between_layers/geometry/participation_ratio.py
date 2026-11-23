"""Participation ratio of the singular-value spectrum."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor, nn

EPS = 1e-12


class ParticipationRatio:
    """``(sum sigma^2)^2 / sum(sigma^4)`` on the embedding singular values.

    Equivalent (up to scaling) to the participation ratio of the covariance
    eigenvalues. For an N-row identity matrix, all singular values equal one
    and the ratio reduces to ``N``. For a rank-1 matrix, only one squared
    singular value is non-zero and the ratio equals one.
    """

    name = "participation_ratio"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["participation_ratio"]

    def compute(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        del labels, attention_mask, model
        x = embeddings.detach().float()
        if x.ndim != 2:
            x = x.reshape(-1, x.shape[-1])
        sigma = torch.linalg.svdvals(x).clamp(min=0.0)
        sq = sigma**2
        denom = (sq**2).sum()
        if float(denom) <= EPS:
            return {"participation_ratio": float("nan")}
        return {"participation_ratio": float((sq.sum() ** 2 / denom).item())}
