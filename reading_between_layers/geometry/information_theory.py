"""Basic information-theoretic estimators (entropy proxies, MI proxies)."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor, nn

logger = logging.getLogger(__name__)

EPS = 1e-12


def _kozachenko_leonenko_entropy(x: np.ndarray, k: int = 3) -> float:
    """Estimate differential entropy via the Kozachenko-Leonenko k-NN estimator.

    Reference: Kozachenko and Leonenko 1987. Returns entropy in nats. Assumes
    ``x`` has shape ``(n, d)`` with ``n > k``.
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    n, d = x.shape
    if n <= k:
        return float("nan")
    tree = cKDTree(x)
    # Query k+1 because the nearest neighbor is the point itself.
    distances, _ = tree.query(x, k=k + 1, p=2.0)
    radii = distances[:, -1]
    radii = np.maximum(radii, np.finfo(float).tiny)
    log_volume = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0)
    # With L2 distances to the k-th nearest neighbor, the unit-ball volume
    # appears directly; no diameter factor is needed.
    return float(
        -digamma(k) + digamma(n) + log_volume + (d / n) * np.log(radii).sum()
    )


class InformationTheoryBasic:
    """Phase-1 information-theory metric bundle.

    Output keys:
    - ``representation_entropy``: Kozachenko-Leonenko entropy of the
      embeddings. PCA-reduces to ``pca_dim`` (default 50) when the ambient
      dimension is larger to keep the k-NN estimate well behaved.
    - ``class_conditional_entropy``: only computed when labels are supplied;
      otherwise ``NaN``. Per-class entropies are averaged weighted by class
      size.
    - ``mutual_info_input_layer``: phase-1 stub. Returns ``NaN`` always; the
      ``allow_mi`` flag is reserved for future estimator implementations.
    """

    name = "information_theory_basic"
    requires_labels = False
    requires_jacobian = False
    output_keys = [
        "representation_entropy",
        "class_conditional_entropy",
        "mutual_info_input_layer",
    ]

    def __init__(self, k: int = 3, pca_dim: int = 50, allow_mi: bool = False) -> None:
        self.k = k
        self.pca_dim = pca_dim
        self.allow_mi = allow_mi

    def _maybe_pca(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] <= self.pca_dim:
            return x
        try:
            from sklearn.decomposition import PCA

            return PCA(n_components=self.pca_dim, random_state=0).fit_transform(x)
        except ImportError:
            logger.warning("sklearn unavailable; skipping PCA reduction")
            return x

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
        if x.shape[0] <= self.k:
            return {key: float("nan") for key in self.output_keys}
        x_np = x.cpu().numpy()
        reduced = self._maybe_pca(x_np)
        try:
            rep_entropy = _kozachenko_leonenko_entropy(reduced, k=self.k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("KL entropy failed: %s", exc)
            rep_entropy = float("nan")
        if labels is not None and labels.numel() == x.shape[0]:
            y = labels.detach().cpu().numpy()
            total = 0.0
            failed = False
            for cls in np.unique(y):
                mask = y == cls
                if mask.sum() <= self.k:
                    continue
                try:
                    total += (mask.sum() / x.shape[0]) * _kozachenko_leonenko_entropy(
                        reduced[mask], k=self.k
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("class-cond entropy failed for class %s: %s", cls, exc)
                    failed = True
                    break
            class_cond = float(total) if not failed else float("nan")
        else:
            class_cond = float("nan")
        if self.allow_mi:
            logger.warning("mutual_info_input_layer is a phase-1 stub; returning NaN")
        mi = float("nan")
        return {
            "representation_entropy": rep_entropy,
            "class_conditional_entropy": class_cond,
            "mutual_info_input_layer": mi,
        }
