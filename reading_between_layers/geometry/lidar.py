"""LiDAR (Linear Discriminability At Rank) and NESum estimators.

NESum is the simpler estimator: the exponential of the entropy of the
normalized embedding-covariance eigenvalues, i.e., the effective number of
dimensions occupied by the representation. It reduces to ``N`` for an N-row
identity matrix.

LiDAR (Thilak et al. 2024) was designed for joint-embedding SSL methods.
Adapting it cleanly to NLP requires augmented views; in phase 1 we approximate
the two views of each input by running the model twice with different dropout
seeds. The implementation computes the within-class and between-class
covariances of the dropout views, forms the matrix
``Sigma_b @ (Sigma_w + ridge_eps * I)^{-1}``, and reports the participation
ratio of its singular values as the LiDAR score.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor, nn

logger = logging.getLogger(__name__)

EPS = 1e-12


class NESum:
    """Effective number of dimensions via the entropy of covariance eigenvalues."""

    name = "nesum"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["nesum"]

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
        n = x.shape[0]
        if n < 2:
            return {"nesum": float("nan")}
        gram = x.t() @ x / n
        eigvals = torch.linalg.eigvalsh(gram).clamp(min=EPS)
        p = eigvals / eigvals.sum()
        entropy = -(p * torch.log(p + EPS)).sum()
        return {"nesum": float(torch.exp(entropy).item())}


class LiDAR:
    """LiDAR estimator adapted for NLP via dropout views.

    ``compute`` expects view A as ``embeddings`` and view B passed via the
    ``labels`` slot when its shape matches ``embeddings``. When no second view
    is supplied we fall back to :class:`NESum` because the joint-embedding
    formulation is undefined.
    """

    name = "lidar"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["lidar"]

    def __init__(self, ridge_eps: float = 1e-3, seed: int = 0) -> None:
        self.ridge_eps = ridge_eps
        self.seed = seed

    def compute(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        del attention_mask, model
        view_a = embeddings.detach().float()
        if view_a.ndim != 2:
            view_a = view_a.reshape(-1, view_a.shape[-1])
        if labels is None or not torch.is_tensor(labels) or labels.shape != view_a.shape:
            nesum = NESum().compute(view_a)
            return {"lidar": nesum["nesum"]}
        view_b = labels.detach().float()
        mean_a = view_a.mean(dim=0, keepdim=True)
        mean_b = view_b.mean(dim=0, keepdim=True)
        centered_a = view_a - mean_a
        centered_b = view_b - mean_b
        between = (mean_a - mean_b).t() @ (mean_a - mean_b)
        n = view_a.shape[0]
        within = (centered_a.t() @ centered_a + centered_b.t() @ centered_b) / max(1, 2 * n - 2)
        eye = torch.eye(within.shape[0], dtype=within.dtype, device=within.device)
        ridge = within + self.ridge_eps * eye
        try:
            solved = torch.linalg.solve(ridge, between)
        except RuntimeError as exc:
            logger.warning("LiDAR solve failed (%s); falling back to NESum", exc)
            return {"lidar": NESum().compute(view_a)["nesum"]}
        sigma = torch.linalg.svdvals(solved).clamp(min=0.0)
        if sigma.numel() == 0:
            return {"lidar": float("nan")}
        sq = sigma**2
        denom = (sq**2).sum()
        if float(denom) <= EPS:
            return {"lidar": float("nan")}
        return {"lidar": float((sq.sum() ** 2 / denom).item())}
