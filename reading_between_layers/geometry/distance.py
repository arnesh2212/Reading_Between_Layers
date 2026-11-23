"""Distance-based descriptors including the geodesic-to-Euclidean ratio."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor, nn

logger = logging.getLogger(__name__)

EPS = 1e-12


class GeodesicEuclideanRatio:
    """Ratio of geodesic distance (k-NN graph shortest path) to Euclidean.

    The geodesic distance is computed from a symmetric k-nearest-neighbor
    graph weighted by Euclidean distances. Reported keys are the mean ratio
    and the standard deviation across pairs.
    """

    name = "geodesic_euclidean_ratio"
    requires_labels = False
    requires_jacobian = False
    output_keys = ["geodesic_euclidean_mean_ratio", "geodesic_euclidean_std_ratio"]

    def __init__(self, k: int = 10, max_pairs: int = 2_000, seed: int = 0) -> None:
        self.k = k
        self.max_pairs = max_pairs
        self.seed = seed

    def compute(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        del labels, attention_mask, model
        try:
            from scipy.sparse.csgraph import shortest_path
            from sklearn.neighbors import kneighbors_graph
        except ImportError:
            return {
                "geodesic_euclidean_mean_ratio": float("nan"),
                "geodesic_euclidean_std_ratio": float("nan"),
            }
        x = embeddings.detach().float()
        if x.ndim != 2:
            x = x.reshape(-1, x.shape[-1])
        n = x.shape[0]
        if n < self.k + 2:
            return {
                "geodesic_euclidean_mean_ratio": float("nan"),
                "geodesic_euclidean_std_ratio": float("nan"),
            }
        x_np = x.cpu().numpy()
        rng = np.random.default_rng(self.seed)
        sample = min(self.max_pairs, n * (n - 1) // 2)
        idx_a = rng.integers(0, n, size=sample)
        idx_b = rng.integers(0, n, size=sample)
        mask = idx_a != idx_b
        idx_a = idx_a[mask]
        idx_b = idx_b[mask]
        if idx_a.size == 0:
            return {
                "geodesic_euclidean_mean_ratio": float("nan"),
                "geodesic_euclidean_std_ratio": float("nan"),
            }
        try:
            graph = kneighbors_graph(x_np, self.k, mode="distance", include_self=False)
            graph = graph.maximum(graph.transpose())
            sources, source_rows = np.unique(idx_a, return_inverse=True)
            geodesic = shortest_path(graph, directed=False, indices=sources)
            geodesic = np.asarray(geodesic)
            if geodesic.ndim == 1:
                geodesic = geodesic[None, :]
        except Exception as exc:  # noqa: BLE001
            logger.warning("geodesic computation failed: %s", exc)
            return {
                "geodesic_euclidean_mean_ratio": float("nan"),
                "geodesic_euclidean_std_ratio": float("nan"),
            }
        ratios = []
        for source_row, i, j in zip(source_rows, idx_a, idx_b, strict=True):
            g = geodesic[source_row, j]
            if not np.isfinite(g) or g <= EPS:
                continue
            e = float(np.linalg.norm(x_np[i] - x_np[j]))
            if e <= EPS:
                continue
            ratios.append(g / e)
        if not ratios:
            return {
                "geodesic_euclidean_mean_ratio": float("nan"),
                "geodesic_euclidean_std_ratio": float("nan"),
            }
        arr = np.asarray(ratios, dtype=np.float64)
        return {
            "geodesic_euclidean_mean_ratio": float(arr.mean()),
            "geodesic_euclidean_std_ratio": float(arr.std()),
        }
