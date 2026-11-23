"""Anisotropy via expected cosine similarity over random pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor, nn

EPS = 1e-12


class AnisotropyRandomPairs:
    """Mean cosine similarity over ``N`` random ordered pairs of embeddings.

    Values close to 0 indicate isotropic representations (random Gaussian rows
    in high dimension yield near-orthogonal pairs). Values close to 1 indicate
    strong anisotropy (most embeddings collinear).
    """

    name = "anisotropy_random_pairs"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["anisotropy_random_pairs"]

    def __init__(self, num_pairs: int = 100_000, seed: int = 0) -> None:
        self.num_pairs = num_pairs
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
        n = x.shape[0]
        if n < 2:
            return {"anisotropy_random_pairs": float("nan")}
        gen = torch.Generator(device=x.device).manual_seed(self.seed)
        idx_a = torch.randint(0, n, (self.num_pairs,), generator=gen, device=x.device)
        idx_b = torch.randint(0, n, (self.num_pairs,), generator=gen, device=x.device)
        keep = idx_a != idx_b
        idx_a = idx_a[keep]
        idx_b = idx_b[keep]
        if len(idx_a) == 0:
            return {"anisotropy_random_pairs": float("nan")}
        a = x[idx_a]
        b = x[idx_b]
        a = a / (a.norm(dim=-1, keepdim=True) + EPS)
        b = b / (b.norm(dim=-1, keepdim=True) + EPS)
        return {"anisotropy_random_pairs": float((a * b).sum(dim=-1).mean().item())}
