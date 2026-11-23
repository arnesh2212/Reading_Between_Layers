"""Intrinsic-dimension estimators for representation point clouds."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor, nn

EPS = 1e-12


class TwoNNIntrinsicDimension:
    """Two-nearest-neighbor intrinsic dimension estimator.

    The estimator fits ``-log(1 - F(mu)) = d * log(mu)`` where
    ``mu = r_2 / r_1`` is the ratio of second-nearest to nearest-neighbor
    distance. Distances are computed in chunks to avoid materializing the full
    pairwise distance matrix.
    """

    name = "twonn_intrinsic_dim"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["twonn_intrinsic_dim"]

    def __init__(self, discard_fraction: float = 0.1, chunk_size: int = 1024) -> None:
        self.discard_fraction = discard_fraction
        self.chunk_size = chunk_size

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
        n = int(x.shape[0])
        if n < 4:
            return {"twonn_intrinsic_dim": float("nan")}

        r1, r2 = _two_nearest_distances(x, chunk_size=self.chunk_size)
        valid = torch.isfinite(r1) & torch.isfinite(r2) & (r1 > EPS) & (r2 > r1)
        if int(valid.sum()) < 4:
            return {"twonn_intrinsic_dim": float("nan")}
        mu = (r2[valid] / r1[valid]).clamp(min=1.0 + EPS)
        log_mu = torch.log(mu)
        log_mu = log_mu[torch.isfinite(log_mu) & (log_mu > EPS)]
        if int(log_mu.numel()) < 4:
            return {"twonn_intrinsic_dim": float("nan")}

        sorted_log_mu, _ = torch.sort(log_mu)
        total = int(sorted_log_mu.numel())
        keep = max(3, int(math.floor(total * (1.0 - self.discard_fraction))))
        keep = min(keep, total - 1) if total > 3 else total
        if keep < 3:
            return {"twonn_intrinsic_dim": float("nan")}

        fit_x = sorted_log_mu[:keep].double()
        cdf = torch.arange(1, keep + 1, dtype=torch.float64, device=fit_x.device) / total
        fit_y = -torch.log1p(-cdf.clamp(max=1.0 - EPS))
        denom = (fit_x * fit_x).sum()
        if float(denom) <= EPS:
            return {"twonn_intrinsic_dim": float("nan")}
        dim = (fit_x * fit_y).sum() / denom
        return {"twonn_intrinsic_dim": float(dim.item())}


def _two_nearest_distances(x: Tensor, chunk_size: int) -> tuple[Tensor, Tensor]:
    n = int(x.shape[0])
    chunk = max(1, int(chunk_size))
    r1_parts: list[Tensor] = []
    r2_parts: list[Tensor] = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        distances = torch.cdist(x[start:end], x)
        rows = torch.arange(end - start, device=x.device)
        cols = torch.arange(start, end, device=x.device)
        distances[rows, cols] = float("inf")
        nearest = torch.topk(distances, k=2, largest=False, dim=1).values
        r1_parts.append(nearest[:, 0])
        r2_parts.append(nearest[:, 1])
    return torch.cat(r1_parts), torch.cat(r2_parts)
