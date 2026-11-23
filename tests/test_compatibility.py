"""Tests for the model-task compatibility matrix and check_compatibility."""

from __future__ import annotations

import pytest

from reading_between_layers.exceptions import UnsupportedModelTaskCombo
from reading_between_layers.models.registry import ModelSpec
from reading_between_layers.tasks.compatibility import TaskType, check_compatibility


def _model(family: str) -> ModelSpec:
    return ModelSpec(
        name=f"fake_{family}",
        hf_id=f"fake/{family}",
        family=family,  # type: ignore[arg-type]
        size_tier="A",
        hidden_size=16,
        num_layers=2,
        has_cls=True,
        supports_chat_template=False,
        pretraining_objective="test",
        pretraining_data_tag="test",
    )


def test_encoder_classification_passes():
    check_compatibility(_model("encoder"), TaskType.CLASSIFICATION)


def test_decoder_retrieval_hard_banned_even_with_override():
    with pytest.raises(UnsupportedModelTaskCombo):
        check_compatibility(_model("decoder"), TaskType.RETRIEVAL)
    with pytest.raises(UnsupportedModelTaskCombo):
        check_compatibility(_model("decoder"), TaskType.RETRIEVAL, override_decoder=True)


def test_decoder_classification_requires_override():
    with pytest.raises(UnsupportedModelTaskCombo):
        check_compatibility(_model("decoder"), TaskType.CLASSIFICATION)


def test_decoder_classification_passes_with_override():
    check_compatibility(_model("decoder"), TaskType.CLASSIFICATION, override_decoder=True)


def test_decoder_multiple_choice_qa_passes():
    check_compatibility(_model("decoder"), TaskType.MULTIPLE_CHOICE_QA)


def test_ssm_passes_for_every_task():
    for task in TaskType:
        check_compatibility(_model("ssm"), task)


def test_encoder_generation_raises():
    with pytest.raises(UnsupportedModelTaskCombo):
        check_compatibility(_model("encoder"), TaskType.GENERATION)


def test_sentence_transformer_treated_as_encoder():
    check_compatibility(_model("sentence_transformer"), TaskType.RETRIEVAL)
    with pytest.raises(UnsupportedModelTaskCombo):
        check_compatibility(_model("sentence_transformer"), TaskType.GENERATION)
