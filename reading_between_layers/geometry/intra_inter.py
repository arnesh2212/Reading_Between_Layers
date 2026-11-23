"""Intra-class vs inter-class cosine similarity for labeled embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor, nn

logger = logging.getLogger(__name__)

EPS = 1e-12


class IntraInterCosine:
    """Mean cosine similarity within vs between classes.

    Labels are optional. When absent we cluster the embeddings with KMeans
    using ``k = max(2, sqrt(n))`` and treat the cluster ids as pseudo-labels.
    Reports the within-class mean, between-class mean, and their difference.
    """

    name = "intra_inter_cosine"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["intra_cosine_mean", "inter_cosine_mean", "intra_inter_gap"]

    def __init__(
        self,
        num_clusters: int | None = None,
        seed: int = 0,
        exact_threshold: int = 4096,
        max_pairs: int = 200_000,
    ) -> None:
        self.num_clusters = num_clusters
        self.seed = seed
        self.exact_threshold = exact_threshold
        self.max_pairs = max_pairs

    def compute(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        del attention_mask, model
        x = embeddings.detach().float()
        if x.ndim != 2:
            x = x.reshape(-1, x.shape[-1])
        n = x.shape[0]
        if n < 4:
            return {k: float("nan") for k in self.output_keys}
        x = x / (x.norm(dim=-1, keepdim=True) + EPS)
        if labels is None:
            try:
                from sklearn.cluster import KMeans, MiniBatchKMeans

                k = self.num_clusters or max(2, int(round(n**0.5)))
                x_np = x.cpu().numpy()
                if n > self.exact_threshold:
                    km = MiniBatchKMeans(
                        n_clusters=k,
                        batch_size=min(4096, n),
                        n_init=3,
                        random_state=self.seed,
                    )
                else:
                    km = KMeans(n_clusters=k, n_init=10, random_state=self.seed)
                lbl = torch.from_numpy(km.fit_predict(x_np)).to(x.device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("kmeans failed: %s", exc)
                return {k: float("nan") for k in self.output_keys}
        else:
            lbl = labels.detach().to(x.device)
        if n > self.exact_threshold:
            return self._sampled_gap(x, lbl)
        sim = x @ x.t()
        same = lbl.unsqueeze(0) == lbl.unsqueeze(1)
        eye = torch.eye(n, dtype=torch.bool, device=x.device)
        intra_mask = same & ~eye
        inter_mask = ~same & ~eye
        intra_count = int(intra_mask.sum().item())
        inter_count = int(inter_mask.sum().item())
        intra = float(sim[intra_mask].mean().item()) if intra_count > 0 else float("nan")
        inter = float(sim[inter_mask].mean().item()) if inter_count > 0 else float("nan")
        gap = intra - inter if intra == intra and inter == inter else float("nan")
        return {
            "intra_cosine_mean": intra,
            "inter_cosine_mean": inter,
            "intra_inter_gap": gap,
        }

    def _sampled_gap(self, x: Tensor, labels: Tensor) -> dict[str, float]:
        """Estimate intra/inter means from random pairs to avoid O(N^2) memory."""
        n = x.shape[0]
        if n < 2:
            return {k: float("nan") for k in self.output_keys}
        gen = torch.Generator(device=x.device).manual_seed(self.seed)
        idx_a = torch.randint(0, n, (self.max_pairs,), generator=gen, device=x.device)
        idx_b = torch.randint(0, n, (self.max_pairs,), generator=gen, device=x.device)
        keep = idx_a != idx_b
        idx_a = idx_a[keep]
        idx_b = idx_b[keep]
        if idx_a.numel() == 0:
            return {k: float("nan") for k in self.output_keys}
        sims = (x[idx_a] * x[idx_b]).sum(dim=-1)
        same = labels[idx_a] == labels[idx_b]
        intra = float(sims[same].mean().item()) if bool(same.any()) else float("nan")
        inter_mask = ~same
        inter = float(sims[inter_mask].mean().item()) if bool(inter_mask.any()) else float("nan")
        gap = intra - inter if intra == intra and inter == inter else float("nan")
        return {
            "intra_cosine_mean": intra,
            "inter_cosine_mean": inter,
            "intra_inter_gap": gap,
        }
