"""Task-to-model-family compatibility matrix and the check_compatibility entry point."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from ..exceptions import UnsupportedModelTaskCombo

if TYPE_CHECKING:
    from ..models.registry import ModelSpec


class TaskType(StrEnum):
    """Enumeration of every supported probing or generation task type."""

    CLASSIFICATION = "classification"
    TOKEN_CLASSIFICATION = "token_classification"
    PAIR_CLASSIFICATION = "pair_classification"
    STS = "sts"
    RETRIEVAL = "retrieval"
    CLUSTERING = "clustering"
    MULTIPLE_CHOICE_QA = "multiple_choice_qa"
    QA_EXTRACTIVE = "qa_extractive"
    QA_GENERATIVE = "qa_generative"
    GENERATION = "generation"


CompatRule = bool | Literal["optional", "no"]


COMPATIBILITY: dict[TaskType, dict[str, CompatRule]] = {
    TaskType.CLASSIFICATION: {
        "encoder": True,
        "decoder": "optional",
        "encoder_decoder": True,
        "ssm": True,
    },
    TaskType.TOKEN_CLASSIFICATION: {
        "encoder": True,
        "decoder": "optional",
        "encoder_decoder": True,
        "ssm": True,
    },
    TaskType.PAIR_CLASSIFICATION: {
        "encoder": True,
        "decoder": "optional",
        "encoder_decoder": True,
        "ssm": True,
    },
    TaskType.STS: {
        "encoder": True,
        "decoder": "optional",
        "encoder_decoder": True,
        "ssm": True,
    },
    TaskType.RETRIEVAL: {
        "encoder": True,
        "decoder": "no",
        "encoder_decoder": True,
        "ssm": True,
    },
    TaskType.CLUSTERING: {
        "encoder": True,
        "decoder": "optional",
        "encoder_decoder": True,
        "ssm": True,
    },
    TaskType.MULTIPLE_CHOICE_QA: {
        "encoder": True,
        "decoder": True,
        "encoder_decoder": True,
        "ssm": True,
    },
    TaskType.QA_EXTRACTIVE: {
        "encoder": True,
        "decoder": "optional",
        "encoder_decoder": True,
        "ssm": True,
    },
    TaskType.QA_GENERATIVE: {
        "encoder": False,
        "decoder": True,
        "encoder_decoder": True,
        "ssm": True,
    },
    TaskType.GENERATION: {
        "encoder": False,
        "decoder": True,
        "encoder_decoder": True,
        "ssm": True,
    },
}


def check_compatibility(
    model_spec: ModelSpec,
    task_type: TaskType,
    override_decoder: bool = False,
) -> None:
    """Raise if the given (model family, task type) pair is disallowed.

    Sentence-transformer models are treated as encoders for compatibility.
    Retrieval combined with a decoder is a hard ban: ``override_decoder`` does
    not unlock it.

    Args:
        model_spec: registry entry for the model.
        task_type: the task being run.
        override_decoder: opt-in flag that unlocks decoder family for tasks
            marked ``"optional"``.

    Raises:
        UnsupportedModelTaskCombo: if the combination is disallowed.
    """
    family = "encoder" if model_spec.family == "sentence_transformer" else model_spec.family
    rule = COMPATIBILITY[task_type].get(family)
    if rule is True:
        return
    if rule == "optional":
        if family == "decoder" and override_decoder:
            return
        raise UnsupportedModelTaskCombo(
            f"{model_spec.name} ({family}) on {task_type.value} requires override_decoder=True"
        )
    if rule == "no":
        raise UnsupportedModelTaskCombo(
            f"{model_spec.name} ({family}) is hard-banned for {task_type.value}"
        )
    raise UnsupportedModelTaskCombo(
        f"{model_spec.name} ({family}) is not allowed for {task_type.value}"
    )
