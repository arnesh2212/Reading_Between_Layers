"""Shared probe trainer consistent across all probe types."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ..tasks.compatibility import TaskType

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from .base import ProbeBase

logger = logging.getLogger(__name__)


class _TaskLike(Protocol):
    def evaluate(self, predictions: Tensor, labels: Tensor) -> dict[str, float]: ...


class _BatchLoader(Protocol):
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]: ...

    def __len__(self) -> int: ...


def _loss_fn_for(task_type: TaskType, label_smoothing: float = 0.0) -> nn.Module:
    """Pick the loss given a task type."""
    if task_type in (
        TaskType.CLASSIFICATION,
        TaskType.PAIR_CLASSIFICATION,
        TaskType.TOKEN_CLASSIFICATION,
        TaskType.MULTIPLE_CHOICE_QA,
    ):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if task_type == TaskType.STS:
        return nn.MSELoss()
    raise NotImplementedError(f"loss for {task_type.value} not implemented in v1")


def _resolve_device(requested: str | torch.device | None) -> torch.device:
    """Resolve the probe-training device and fail clearly for unavailable CUDA."""
    if requested is None:
        return torch.device("cpu")
    device = torch.device(str(requested))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("probe training requested cuda, but torch.cuda.is_available() is False")
    return device


def _predict(probe: ProbeBase, embeddings: Tensor) -> Tensor:
    """Pure-eval forward: returns logits (classification) or scalar (regression)."""
    probe.eval()
    with torch.no_grad():
        return probe(embeddings)


def _predict_batched(
    probe: ProbeBase,
    embeddings: Tensor,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Run eval forward in bounded batches and return CPU logits."""
    if embeddings.size(0) == 0:
        return _predict(probe, embeddings.to(device)).detach().cpu()
    outputs: list[Tensor] = []
    non_blocking = device.type == "cuda"
    for start in range(0, embeddings.size(0), batch_size):
        batch = embeddings[start : start + batch_size].to(device, non_blocking=non_blocking)
        outputs.append(_predict(probe, batch).detach().cpu())
    return torch.cat(outputs, dim=0)


def _decode(logits: Tensor, task_type: TaskType) -> Tensor:
    if task_type == TaskType.STS:
        return logits.squeeze(-1)
    if task_type == TaskType.MULTIPLE_CHOICE_QA:
        return _multiple_choice_logits(logits).argmax(dim=-1)
    return logits.argmax(dim=-1)


def _multiple_choice_logits(logits: Tensor) -> Tensor:
    if logits.dim() == 3 and logits.size(-1) == 1:
        return logits.squeeze(-1)
    return logits


def _linear_warmup_lambda(total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return fn


def _coerce_labels(labels: Tensor, task_type: TaskType) -> Tensor:
    labels = labels.detach().cpu()
    if task_type == TaskType.STS:
        return labels.float()
    return labels.long()


def _move_features(batch_x: Tensor, device: torch.device) -> Tensor:
    return batch_x.to(
        device=device,
        dtype=torch.float32,
        non_blocking=device.type == "cuda",
    )


def _move_labels(batch_y: Tensor, task_type: TaskType, device: torch.device) -> Tensor:
    batch_y = batch_y.to(device=device, non_blocking=device.type == "cuda")
    if task_type == TaskType.STS:
        return batch_y.float()
    return batch_y.long()


def _predict_loader(
    probe: ProbeBase,
    loader: _BatchLoader,
    task_type: TaskType,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    outputs: list[Tensor] = []
    labels: list[Tensor] = []
    for batch_x, batch_y in loader:
        batch_x = _move_features(batch_x, device)
        outputs.append(_predict(probe, batch_x).detach().cpu())
        labels.append(_coerce_labels(batch_y, task_type))
    if not outputs:
        raise ValueError("cannot evaluate probe on an empty loader")
    return torch.cat(outputs, dim=0), torch.cat(labels, dim=0)


def _bootstrap_ci(values: np.ndarray, seed: int, n_resamples: int = 1000) -> tuple[float, float]:
    """Return 2.5/97.5 percentile bootstrap CI of the per-example values."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(values[idx].mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _selection_metric_name(config: DictConfig, metrics: dict[str, float]) -> str:
    evaluation = config.get("evaluation", {})
    requested = str(evaluation.get("selection_metric", "primary"))
    if requested == "primary":
        return next(iter(metrics))
    if requested == "macro_f1_if_available":
        return "macro_f1" if "macro_f1" in metrics else next(iter(metrics))
    if requested not in metrics:
        available = ", ".join(metrics)
        raise ValueError(f"selection_metric={requested!r} not in metrics: {available}")
    return requested


def train_probe(
    embeddings: Tensor,
    labels: Tensor,
    val_embeddings: Tensor,
    val_labels: Tensor,
    test_embeddings: Tensor,
    test_labels: Tensor,
    probe: ProbeBase,
    task: _TaskLike,
    config: DictConfig,
    seed: int,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train ``probe`` with early stopping and return metrics with bootstrap CIs."""
    task_type = probe.task_type
    label_tensor = _coerce_labels(labels, task_type)
    val_label_tensor = _coerce_labels(val_labels, task_type)
    test_label_tensor = _coerce_labels(test_labels, task_type)
    batch_size = int(config.training.batch_size)
    eval_batch_size = int(config.training.get("eval_batch_size", batch_size))
    requested_device = device if device is not None else config.get("device", None)
    train_device = _resolve_device(requested_device)

    train_loader = DataLoader(
        TensorDataset(embeddings.detach().cpu(), label_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=train_device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(val_embeddings.detach().cpu(), val_label_tensor),
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=train_device.type == "cuda",
    )
    test_loader = DataLoader(
        TensorDataset(test_embeddings.detach().cpu(), test_label_tensor),
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=train_device.type == "cuda",
    )
    return train_probe_from_loaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        probe=probe,
        task=task,
        config=config,
        seed=seed,
        device=train_device,
    )


def train_probe_from_loaders(
    train_loader: _BatchLoader,
    val_loader: _BatchLoader,
    test_loader: _BatchLoader,
    probe: ProbeBase,
    task: _TaskLike,
    config: DictConfig,
    seed: int,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train ``probe`` from batch loaders without materializing full embeddings."""
    torch.manual_seed(seed)
    requested_device = device if device is not None else config.get("device", None)
    train_device = _resolve_device(requested_device)
    if train_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    task_type = probe.task_type
    probe = probe.to(train_device)
    label_smoothing = float(config.get("loss", {}).get("label_smoothing", 0.0))
    loss_fn = _loss_fn_for(task_type, label_smoothing=label_smoothing).to(train_device)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(config.optimizer.lr),
        weight_decay=float(config.optimizer.weight_decay),
    )

    max_epochs = int(config.training.max_epochs)
    total_steps = max(1, len(train_loader) * max_epochs)
    warmup_ratio = float(config.scheduler.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_linear_warmup_lambda(total_steps, warmup_ratio)
    )
    grad_clip = float(config.training.gradient_clip)
    patience = int(config.training.early_stopping_patience)

    best_val_metric = -float("inf")
    best_state: dict | None = None
    bad_epochs = 0

    for epoch in range(max_epochs):
        probe.train()
        for batch_x, batch_y in train_loader:
            batch_x = _move_features(batch_x, train_device)
            batch_y = _move_labels(batch_y, task_type, train_device)
            optimizer.zero_grad()
            logits = probe(batch_x)
            if task_type == TaskType.STS:
                loss = loss_fn(logits.squeeze(-1), batch_y)
            elif task_type == TaskType.MULTIPLE_CHOICE_QA:
                loss = loss_fn(_multiple_choice_logits(logits), batch_y)
            else:
                loss = loss_fn(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(probe.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

        val_logits, val_label_tensor = _predict_loader(
            probe,
            val_loader,
            task_type,
            train_device,
        )
        val_preds = _decode(val_logits, task_type)
        val_metrics = task.evaluate(val_preds, val_label_tensor)
        selection_metric = _selection_metric_name(config, val_metrics)
        primary = float(val_metrics[selection_metric])
        if primary > best_val_metric:
            best_val_metric = primary
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("early stopping at epoch %d (patience %d)", epoch, patience)
                break

    if best_state is not None:
        probe.load_state_dict(best_state)

    test_logits, test_label_tensor = _predict_loader(
        probe,
        test_loader,
        task_type,
        train_device,
    )
    test_preds = _decode(test_logits, task_type)
    test_metrics = task.evaluate(test_preds, test_label_tensor)

    if task_type == TaskType.STS:
        per_example = (test_preds - test_label_tensor.float()).abs().cpu().numpy()
        primary_name = next(iter(test_metrics))
        ci_low, ci_high = _bootstrap_ci(-per_example, seed)  # higher is better
    else:
        per_example = (test_preds == test_label_tensor).float().cpu().numpy()
        primary_name = next(iter(test_metrics))
        ci_low, ci_high = _bootstrap_ci(per_example, seed)

    return {
        "seed": seed,
        "val_primary": float(best_val_metric),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "test_primary_ci_lo": float(ci_low),
        "test_primary_ci_hi": float(ci_high),
        "primary_name": primary_name,
        "num_params": probe.num_params,
    }
