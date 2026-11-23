"""Linear probe: a single ``nn.Linear`` on frozen pooled embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from .base import ProbeBase

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from ..tasks.compatibility import TaskType


class LinearProbe(ProbeBase):
    """Single linear layer mapping ``input_dim`` to ``output_dim``."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: TaskType,
        config: DictConfig,
    ) -> None:
        super().__init__(input_dim, output_dim, task_type, config)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.linear(embeddings)
