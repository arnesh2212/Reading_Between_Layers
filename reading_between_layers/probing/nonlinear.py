"""Nonlinear probe: ``Linear -> GELU -> Linear`` on frozen pooled embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from .base import ProbeBase

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from ..tasks.compatibility import TaskType


class NonlinearProbe(ProbeBase):
    """Two-layer MLP head with hidden size ``round(input_dim * hidden_size_ratio)``."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: TaskType,
        config: DictConfig,
    ) -> None:
        super().__init__(input_dim, output_dim, task_type, config)
        ratio = float(getattr(config, "hidden_size_ratio", 0.5))
        hidden = max(1, int(input_dim * ratio))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.net(embeddings)
