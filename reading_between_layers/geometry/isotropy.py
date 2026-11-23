"""Isotropy metrics including the Mu / Viswanath measure."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor, nn

logger = logging.getLogger(__name__)

EPS = 1e-12


class IsotropyMuViswanath:
    """Partition-function isotropy proxy from Mu and Viswanath 2018.

    Computes ``Z(c) = sum_n exp(c^T x_n)`` for many random unit-norm directions
    ``c``. The reported score is ``min Z / max Z``: 1.0 indicates perfect
    isotropy and values closer to 0 indicate strong anisotropy. We project
    onto eigenvectors of the empirical second-moment matrix to reduce variance,
    matching the original recipe.
    """

    name = "isotropy_mu_viswanath"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["iso_mu_viswanath"]

    def __init__(self, num_directions: int = 32, seed: int = 0) -> None:
        self.num_directions = num_directions
        self.seed = seed

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
        gen = torch.Generator(device=x.device).manual_seed(self.seed)
        d = x.shape[-1]
        # Use eigenvectors of XtX as candidate directions when ``num_directions``
        # exceeds d; otherwise sample random unit vectors.
        if self.num_directions <= d:
            try:
                _, eigvecs = torch.linalg.eigh(x.t() @ x)
                directions = eigvecs[:, -self.num_directions :].t()
            except Exception as exc:  # noqa: BLE001
                logger.warning("eigh failed (%s); falling back to random dirs", exc)
                directions = torch.randn(
                    self.num_directions,
                    d,
                    generator=gen,
                    device=x.device,
                )
        else:
            directions = torch.randn(self.num_directions, d, generator=gen, device=x.device)
        directions = directions / (directions.norm(dim=-1, keepdim=True) + EPS)
        partition = torch.logsumexp(directions @ x.t(), dim=-1)
        return {
            "iso_mu_viswanath": float(
                torch.exp(partition.min() - partition.max()).clamp(0.0, 1.0).item()
            )
        }


class IsotropyEigenvalueRatio:
    """Mean over standard-deviation of the second-moment-matrix eigenvalues.

    Defined as ``mu / sqrt(var(lambda) + eps)`` where ``lambda`` are the
    eigenvalues of ``X^T X / n``. Higher values mean flatter spectra and so
    closer to isotropic.
    """

    name = "isotropy_eigenvalue_ratio"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["iso_eigenvalue_ratio"]

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
        if n <= 1:
            return {"iso_eigenvalue_ratio": float("nan")}
        gram = x.t() @ x / n
        eigvals = torch.linalg.eigvalsh(gram).clamp(min=EPS)
        mu = float(eigvals.mean())
        std = float(eigvals.std(unbiased=False))
        return {"iso_eigenvalue_ratio": mu / (std + EPS)}
