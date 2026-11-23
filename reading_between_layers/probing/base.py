"""Abstract probe base class shared by linear, nonlinear, LoRA, and full FT variants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from ..tasks.compatibility import TaskType


class ProbeBase(nn.Module, ABC):
    """Common interface for probe heads over frozen pooled embeddings."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: TaskType,
        config: DictConfig,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.config = config

    @abstractmethod
    def forward(self, embeddings: Tensor) -> Tensor:
        """Map pooled embeddings to task logits or scores."""

    @property
    def num_params(self) -> int:
        """Total parameter count across this probe."""
        return sum(p.numel() for p in self.parameters())
