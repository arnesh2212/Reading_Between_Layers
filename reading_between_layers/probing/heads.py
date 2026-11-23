"""Task-specific head modules used by probes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from ..tasks.compatibility import TaskType

if TYPE_CHECKING:
    from omegaconf import DictConfig


def make_head(
    task_type: TaskType,
    input_dim: int,
    num_classes: int | None,
    config: DictConfig,
) -> nn.Module:
    """Return the right head module for the requested task type.

    For classification-like tasks the head is a single ``nn.Linear`` projecting
    to ``num_classes``. For STS the head is a single scalar regression head.
    For retrieval the head is an identity-style projection used by the
    bi-encoder loss. Other task types are deferred to phase 2.
    """
    del config
    if task_type in (
        TaskType.CLASSIFICATION,
        TaskType.PAIR_CLASSIFICATION,
        TaskType.TOKEN_CLASSIFICATION,
        TaskType.MULTIPLE_CHOICE_QA,
    ):
        if num_classes is None:
            raise ValueError(f"{task_type.value} requires num_classes")
        return nn.Linear(input_dim, num_classes)
    if task_type == TaskType.STS:
        return nn.Linear(input_dim, 1)
    if task_type == TaskType.RETRIEVAL:
        return nn.Linear(input_dim, input_dim)
    raise NotImplementedError(f"head for {task_type.value} not implemented in v1")
