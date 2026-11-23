"""Representational similarity metrics between activation matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor, nn

EPS = 1e-12


class LinearCKA:
    """Linear centered kernel alignment between two activation matrices."""

    name = "linear_cka"
    requires_labels = True
    requires_jacobian = False
    output_keys = ["linear_cka"]

    def compute(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        del attention_mask, model
        if labels is None or not torch.is_tensor(labels):
            return {"linear_cka": float("nan")}
        if labels.ndim < 2:
            return {"linear_cka": float("nan")}
        x = _as_2d_float(embeddings)
        y = _as_2d_float(labels)
        if x.shape[0] != y.shape[0]:
            n = min(int(x.shape[0]), int(y.shape[0]))
            x = x[:n]
            y = y[:n]
        if x.shape[0] < 2:
            return {"linear_cka": float("nan")}
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        cross = y.t() @ x
        numerator = (cross * cross).sum()
        x_norm = torch.linalg.matrix_norm(x.t() @ x, ord="fro")
        y_norm = torch.linalg.matrix_norm(y.t() @ y, ord="fro")
        denom = x_norm * y_norm
        if float(denom) <= EPS:
            return {"linear_cka": float("nan")}
        value = (numerator / denom).clamp(0.0, 1.0)
        return {"linear_cka": float(value.item())}


def _as_2d_float(tensor: Tensor) -> Tensor:
    out = tensor.detach().float()
    if out.ndim < 2:
        return out.reshape(out.shape[0], 1)
    if out.ndim != 2:
        out = out.reshape(-1, out.shape[-1])
    return out
