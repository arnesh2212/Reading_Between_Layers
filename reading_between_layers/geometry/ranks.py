"""Effective rank, stable rank, and rank-based descriptors of the spectrum."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor, nn

EPS = 1e-12


def _singular_values(x: Tensor) -> Tensor:
    """Compute non-negative singular values of ``x`` clamped at ``EPS``."""
    if x.ndim != 2:
        x = x.reshape(-1, x.shape[-1])
    return torch.linalg.svdvals(x).clamp(min=EPS)


class EffectiveRank:
    """``exp(-sum p_i log p_i)`` with ``p_i = sigma_i / sum(sigma)``."""

    name = "effective_rank"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["effective_rank"]

    def compute(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        del labels, attention_mask, model
        sigma = _singular_values(embeddings.detach().float())
        p = sigma / (sigma.sum() + EPS)
        entropy = -(p * torch.log(p + EPS)).sum()
        return {"effective_rank": float(torch.exp(entropy).item())}


class StableRank:
    """``||A||_F^2 / ||A||_2^2`` = ``sum(sigma_i^2) / sigma_max^2``."""

    name = "stable_rank"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["stable_rank"]

    def compute(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        del labels, attention_mask, model
        sigma = _singular_values(embeddings.detach().float())
        if sigma.numel() == 0:
            return {"stable_rank": float("nan")}
        sigma_max = sigma.max()
        return {"stable_rank": float(((sigma**2).sum() / (sigma_max**2 + EPS)).item())}


class RankMe:
    """Smoothed effective rank (Garrido et al. 2022).

    The Garrido paper adds an epsilon before normalizing to keep zero-singular
    -value entries from dominating. The formula is otherwise identical to the
    classical effective rank.
    """

    name = "rankme"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["rankme"]

    def __init__(self, eps: float = 1e-7) -> None:
        self.eps = eps

    def compute(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        del labels, attention_mask, model
        sigma = _singular_values(embeddings.detach().float())
        sigma = sigma + self.eps
        p = sigma / sigma.sum()
        entropy = -(p * torch.log(p + EPS)).sum()
        return {"rankme": float(torch.exp(entropy).item())}
