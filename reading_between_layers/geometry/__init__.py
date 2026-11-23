"""Geometry and information-theory metrics over per-layer representations."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from .registry import GEOMETRY_REGISTRY

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import Tensor, nn

logger = logging.getLogger(__name__)
EPS = 1e-12

_SHARED_SPECTRAL_METRICS = {
    "effective_rank",
    "stable_rank",
    "participation_ratio",
    "rankme",
    "alpha_req",
    "condition_number",
    "top1_variance_ratio",
}
_SHARED_SECOND_MOMENT_METRICS = {
    "isotropy_mu_viswanath",
    "isotropy_eigenvalue_ratio",
    "lidar",
    "nesum",
}
_SHARED_CENTERED_COV_METRICS = {"isoscore"}
_SHARED_METRICS = (
    _SHARED_SPECTRAL_METRICS
    | _SHARED_SECOND_MOMENT_METRICS
    | _SHARED_CENTERED_COV_METRICS
)


def resolve_geometry_sample_size(total_rows: int, config: DictConfig) -> int:
    """Resolve the number of rows to use from ``sample_size`` or ``sample_fraction``."""
    if total_rows <= 0:
        return 0
    sample_size = config.get("sample_size") if hasattr(config, "get") else None
    if sample_size is not None:
        return max(1, min(total_rows, int(sample_size)))
    sample_fraction = config.get("sample_fraction", 1.0) if hasattr(config, "get") else 1.0
    fraction = float(sample_fraction)
    if fraction <= 0:
        raise ValueError("geometry.sample_fraction must be > 0")
    if fraction >= 1:
        return total_rows
    return max(1, min(total_rows, math.floor(total_rows * fraction)))


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc)
    )


def _as_2d_float(embeddings: Tensor) -> Tensor:
    x = embeddings.detach().float()
    if x.ndim != 2:
        x = x.reshape(-1, x.shape[-1])
    return x


def _alpha_req_from_sigma(sigma: Tensor) -> dict[str, float]:
    if sigma.numel() < 3:
        return {"alpha_req": float("nan"), "alpha_req_r2": float("nan")}
    sorted_sigma, _ = torch.sort(sigma.clamp(min=EPS), descending=True)
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


def _entropy_effective_count(values: Tensor) -> float:
    p = values / (values.sum() + EPS)
    entropy = -(p * torch.log(p + EPS)).sum()
    return float(torch.exp(entropy).item())


def _compute_shared_spectral(x: Tensor, requested: set[str]) -> dict[str, dict[str, float]]:
    if not requested:
        return {}
    sigma_raw = torch.linalg.svdvals(x).clamp(min=0.0)
    sigma = sigma_raw.clamp(min=EPS)
    outputs: dict[str, dict[str, float]] = {}
    if "effective_rank" in requested:
        outputs["effective_rank"] = {"effective_rank": _entropy_effective_count(sigma)}
    if "stable_rank" in requested:
        if sigma.numel() == 0:
            outputs["stable_rank"] = {"stable_rank": float("nan")}
        else:
            sigma_max = sigma.max()
            outputs["stable_rank"] = {
                "stable_rank": float(((sigma**2).sum() / (sigma_max**2 + EPS)).item())
            }
    if "participation_ratio" in requested:
        sq = sigma_raw**2
        denom = (sq**2).sum()
        outputs["participation_ratio"] = {
            "participation_ratio": (
                float((sq.sum() ** 2 / denom).item())
                if float(denom) > EPS
                else float("nan")
            )
        }
    if "rankme" in requested:
        outputs["rankme"] = {"rankme": _entropy_effective_count(sigma + 1e-7)}
    if "alpha_req" in requested:
        outputs["alpha_req"] = _alpha_req_from_sigma(sigma)
    if "condition_number" in requested:
        outputs["condition_number"] = {
            "condition_number": (
                float((sigma.max() / sigma.min()).item())
                if sigma.numel()
                else float("nan")
            )
        }
    if "top1_variance_ratio" in requested:
        sq = sigma_raw**2
        total = sq.sum()
        outputs["top1_variance_ratio"] = {
            "top1_variance_ratio": (
                float((sq.max() / total).item())
                if sigma_raw.numel() and float(total) > EPS
                else float("nan")
            )
        }
    return outputs


def _compute_shared_second_moment(
    x: Tensor,
    requested: set[str],
    labels: Tensor | None,
) -> dict[str, dict[str, float]]:
    if not requested:
        return {}
    n, d = x.shape
    outputs: dict[str, dict[str, float]] = {}
    if n <= 1:
        for metric_name in requested:
            outputs[metric_name] = {
                key: float("nan")
                for key in getattr(GEOMETRY_REGISTRY.get(metric_name), "output_keys", [metric_name])
            }
        return outputs

    need_vectors = "isotropy_mu_viswanath" in requested and d >= 32
    gram = x.t() @ x
    if need_vectors:
        eigvals_raw, eigvecs = torch.linalg.eigh(gram)
    else:
        eigvals_raw = torch.linalg.eigvalsh(gram)
        eigvecs = None
    eigvals = (eigvals_raw / n).clamp(min=EPS)

    if "isotropy_eigenvalue_ratio" in requested:
        mu = float(eigvals.mean())
        std = float(eigvals.std(unbiased=False))
        outputs["isotropy_eigenvalue_ratio"] = {"iso_eigenvalue_ratio": mu / (std + EPS)}

    if "nesum" in requested or "lidar" in requested:
        nesum = _entropy_effective_count(eigvals)
        if "nesum" in requested:
            outputs["nesum"] = {"nesum": nesum}
        if "lidar" in requested:
            if labels is not None and torch.is_tensor(labels) and labels.shape == x.shape:
                requested.discard("lidar")
            else:
                outputs["lidar"] = {"lidar": nesum}

    if "isotropy_mu_viswanath" in requested:
        num_directions = 32
        if num_directions <= d and eigvecs is not None:
            directions = eigvecs[:, -num_directions:].t()
        else:
            gen = torch.Generator(device=x.device).manual_seed(0)
            directions = torch.randn(num_directions, d, generator=gen, device=x.device)
        directions = directions / (directions.norm(dim=-1, keepdim=True) + EPS)
        partition = torch.logsumexp(directions @ x.t(), dim=-1)
        outputs["isotropy_mu_viswanath"] = {
            "iso_mu_viswanath": float(
                torch.exp(partition.min() - partition.max()).clamp(0.0, 1.0).item()
            )
        }
    return outputs


def _compute_shared_centered_cov(
    x: Tensor,
    requested: set[str],
) -> dict[str, dict[str, float]]:
    if not requested:
        return {}
    n, d = x.shape
    outputs: dict[str, dict[str, float]] = {}
    if "isoscore" in requested:
        if n < 2 or d < 2:
            outputs["isoscore"] = {"isoscore": float("nan")}
        else:
            centered = x - x.mean(dim=0, keepdim=True)
            cov = centered.t() @ centered / (n - 1)
            eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
            norm = torch.linalg.norm(eigvals)
            if float(norm) <= EPS:
                score = 0.0
            else:
                normalized = eigvals / norm
                uniform = torch.full_like(normalized, 1.0 / (d**0.5))
                dist_sq = float(((normalized - uniform) ** 2).sum())
                score = max(0.0, 1.0 - dist_sq / 2.0)
            outputs["isoscore"] = {"isoscore": float(score)}
    return outputs


def _compute_shared_geometry(
    x: Tensor,
    labels: Tensor | None,
    requested: set[str],
) -> dict[str, dict[str, float]]:
    """Compute metrics that share SVD/eigen work and return by metric name."""
    shared: dict[str, dict[str, float]] = {}
    spectral = requested & _SHARED_SPECTRAL_METRICS
    second_moment = requested & _SHARED_SECOND_MOMENT_METRICS
    centered_cov = requested & _SHARED_CENTERED_COV_METRICS

    shared.update(_compute_shared_spectral(x, spectral))
    shared.update(_compute_shared_second_moment(x, set(second_moment), labels))
    shared.update(_compute_shared_centered_cov(x, centered_cov))
    return shared


def compute_all_geometry(
    embeddings: Tensor,
    labels: Tensor | None,
    config: DictConfig,
    model: nn.Module | None = None,
) -> dict[str, float]:
    """Run every metric enabled in ``config.metrics`` and merge their outputs.

    Args:
        embeddings: pooled embedding tensor ``(n, d)``.
        labels: optional integer labels used by ``intra_inter_cosine`` and
            ``information_theory_basic.class_conditional_entropy``.
        config: a Hydra geometry config (``cfg.geometry``). Reads ``metrics``
            (mapping name -> bool) and ``sample_size``.
        model: optional model; only consulted by Jacobian-based metrics.

    Returns:
        A flat dict mapping each enabled metric's output key to its float
        value. Failed metrics log a warning and contribute ``NaN`` for every
        output key they declare.
    """
    if embeddings.numel() == 0:
        raise ValueError("compute_all_geometry called with an empty embedding tensor")
    sample_size = resolve_geometry_sample_size(int(embeddings.shape[0]), config)
    x = embeddings
    if 0 < sample_size < x.shape[0]:
        # Deterministic subsampling: leading rows so repeated runs hash the same.
        x = x[:sample_size]
        if labels is not None:
            labels = labels[:sample_size]
    metrics_cfg = dict(config.metrics) if hasattr(config, "metrics") else {}
    requested_shared = {
        name
        for name, enabled in metrics_cfg.items()
        if enabled and name in _SHARED_METRICS
    }
    shared_by_metric: dict[str, dict[str, float]] = {}
    if requested_shared:
        try:
            shared_by_metric = _compute_shared_geometry(_as_2d_float(x), labels, requested_shared)
        except Exception as exc:  # noqa: BLE001
            if _is_cuda_oom(exc):
                raise
            logger.warning("shared geometry decomposition failed; falling back per metric: %s", exc)

    results: dict[str, float] = {}
    for metric_name, enabled in metrics_cfg.items():
        if not enabled:
            continue
        if metric_name in shared_by_metric:
            for key, value in shared_by_metric[metric_name].items():
                results[key] = float(value)
            continue
        cls = GEOMETRY_REGISTRY.get(metric_name)
        if cls is None:
            logger.warning("metric %s not in registry; skipping", metric_name)
            continue
        try:
            instance = cls()
            kwargs: dict = {}
            if getattr(instance, "requires_labels", False) and labels is not None:
                kwargs["labels"] = labels
            elif labels is not None and metric_name in (
                "intra_inter_cosine",
                "information_theory_basic",
            ):
                kwargs["labels"] = labels
            if getattr(instance, "requires_jacobian", False) and model is not None:
                kwargs["model"] = model
            output = instance.compute(x, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if _is_cuda_oom(exc):
                raise
            logger.warning("metric %s failed: %s", metric_name, exc)
            output_keys = getattr(cls, "output_keys", [metric_name])
            output = {k: float("nan") for k in output_keys}
        for key, value in output.items():
            results[key] = float(value)
    return results
