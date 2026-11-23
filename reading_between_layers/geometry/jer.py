"""Joint embedding rank (JER). Heavy compute; gated behind a config flag.

For each sample we compute the Jacobian ``df/dx`` of the model output with
respect to the input embedding and take the participation ratio of its
singular values. Per-sample backward passes make this metric expensive. The
default ``sample_cap`` of 256 keeps it tractable on a 20GB GPU; callers
should reduce it further when running over large models.

The metric is disabled by default in the geometry configs. We always log a
warning when it is enabled.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor, nn

logger = logging.getLogger(__name__)

EPS = 1e-12


class JacobianEffectiveRank:
    """Participation ratio of the Jacobian's singular values."""

    name = "jer"
    requires_labels = False
    requires_jacobian = True
    output_keys = ["jer"]

    def __init__(self, sample_cap: int = 256, seed: int = 0) -> None:
        self.sample_cap = sample_cap
        self.seed = seed

    def compute(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        del labels, attention_mask
        if model is None:
            logger.warning("JER requires a model; returning NaN")
            return {"jer": float("nan")}
        logger.warning(
            "JER enabled: per-sample backward is slow; consider reducing sample_cap=%d",
            self.sample_cap,
        )
        x = embeddings.detach().float()
        if x.ndim != 2:
            x = x.reshape(-1, x.shape[-1])
        n = min(x.shape[0], self.sample_cap)
        rng = torch.Generator(device=x.device).manual_seed(self.seed)
        perm = torch.randperm(x.shape[0], generator=rng, device=x.device)[:n]
        accum: list[Tensor] = []
        for idx in perm.tolist():
            inp = x[idx : idx + 1].clone().requires_grad_(True)
            try:
                out = model(inp)
                if isinstance(out, tuple):
                    out = out[0]
                out = out.reshape(-1)
                jac_rows = []
                for k in range(out.shape[0]):
                    grad = torch.autograd.grad(
                        out[k], inp, retain_graph=k < out.shape[0] - 1, create_graph=False
                    )[0]
                    jac_rows.append(grad.reshape(-1))
                accum.append(torch.stack(jac_rows, dim=0).detach())
            except Exception as exc:  # noqa: BLE001
                logger.warning("JER per-sample failed: %s", exc)
                return {"jer": float("nan")}
        if not accum:
            return {"jer": float("nan")}
        jac = torch.cat(accum, dim=0)
        sigma = torch.linalg.svdvals(jac).clamp(min=0.0)
        sq = sigma**2
        denom = (sq**2).sum()
        if float(denom) <= EPS:
            return {"jer": float("nan")}
        return {"jer": float((sq.sum() ** 2 / denom).item())}
