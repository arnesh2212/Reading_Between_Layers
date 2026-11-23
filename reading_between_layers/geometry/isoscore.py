"""IsoScore measure of approximate isotropy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor, nn

EPS = 1e-12


class IsoScore:
    """IsoScore from Rudman et al. 2022.

    Steps:
    1. Center embeddings.
    2. Diagonalize the covariance via PCA.
    3. Read off the diagonal of the rotated covariance (the eigenvalues).
    4. Normalize the diagonal to have unit Euclidean norm.
    5. Compute the squared L2 distance from the uniform vector
       ``(1/sqrt(d)) * ones(d)``; rescale to ``[0, 1]`` so that an isotropic
       Gaussian gives ~1 and a degenerate (rank-1) distribution gives ~0.
    """

    name = "isoscore"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["isoscore"]

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
        n, d = x.shape
        if n < 2 or d < 2:
            return {"isoscore": float("nan")}
        centered = x - x.mean(dim=0, keepdim=True)
        cov = centered.t() @ centered / (n - 1)
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
        norm = torch.linalg.norm(eigvals)
        if float(norm) <= EPS:
            return {"isoscore": 0.0}
        normalized = eigvals / norm
        uniform = torch.full_like(normalized, 1.0 / (d**0.5))
        # squared distance is in [0, 2], scale so that uniform=>0 (perfect)
        # and one-hot (a single dominant eigenvalue) => 2 (worst).
        dist_sq = float(((normalized - uniform) ** 2).sum())
        # Map distance squared (0 = isotropic, 2 = degenerate) to a score in
        # [0, 1] where 1 = isotropic.
        return {"isoscore": float(max(0.0, 1.0 - dist_sq / 2.0))}
