"""Spectral descriptors from the singular value spectrum."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor, nn

EPS = 1e-12


class AlphaReQ:
    """Power-law decay exponent from a log-log fit of the singular values.

    Fits ``log(sigma_i) ~ -alpha * log(i)`` via ordinary least squares on the
    sorted (descending) singular values. Returns the slope ``alpha`` and the
    fit's coefficient of determination as ``alpha_req_r2``.
    """

    name = "alpha_req"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["alpha_req", "alpha_req_r2"]

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
        sigma = torch.linalg.svdvals(x).clamp(min=EPS)
        if sigma.numel() < 3:
            return {"alpha_req": float("nan"), "alpha_req_r2": float("nan")}
        sorted_sigma, _ = torch.sort(sigma, descending=True)
        idx = torch.arange(
            1,
            sorted_sigma.numel() + 1,
            dtype=torch.float64,
            device=sorted_sigma.device,
        )
        log_idx = torch.log(idx)
        log_sigma = torch.log(sorted_sigma.double())
        mean_x = log_idx.mean()
        mean_y = log_sigma.mean()
        var_x = ((log_idx - mean_x) ** 2).sum()
        cov_xy = ((log_idx - mean_x) * (log_sigma - mean_y)).sum()
        slope = cov_xy / (var_x + EPS)
        intercept = mean_y - slope * mean_x
        predicted = intercept + slope * log_idx
        ss_res = ((log_sigma - predicted) ** 2).sum()
        ss_tot = ((log_sigma - mean_y) ** 2).sum()
        r2 = 1.0 - float(ss_res / (ss_tot + EPS))
        return {"alpha_req": float(-slope.item()), "alpha_req_r2": r2}


class ConditionNumber:
    """Ratio of the largest to smallest singular value."""

    name = "condition_number"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["condition_number"]

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
        sigma = torch.linalg.svdvals(x).clamp(min=EPS)
        if sigma.numel() == 0:
            return {"condition_number": float("nan")}
        return {"condition_number": float((sigma.max() / sigma.min()).item())}


class Top1VarianceRatio:
    """Fraction of squared singular-value mass carried by the top component."""

    name = "top1_variance_ratio"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["top1_variance_ratio"]

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
        if sigma.numel() == 0:
            return {"top1_variance_ratio": float("nan")}
        sq = sigma**2
        total = sq.sum()
        if float(total) <= EPS:
            return {"top1_variance_ratio": float("nan")}
        return {"top1_variance_ratio": float((sq.max() / total).item())}
