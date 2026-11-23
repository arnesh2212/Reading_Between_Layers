"""Tests for the model and dataset registries."""

from __future__ import annotations

from reading_between_layers.datasets.registry import DATASET_REGISTRY, list_datasets_by_language
from reading_between_layers.models.registry import (
    MODEL_REGISTRY,
    list_models_by_family,
    list_models_by_tier,
)
from reading_between_layers.tasks.compatibility import TaskType

VALID_MODEL_FAMILIES = {
    "encoder",
    "decoder",
    "encoder_decoder",
    "ssm",
    "sentence_transformer",
}
VALID_TIERS = {"A", "B", "C"}
VALID_DATASET_TIERS = {"tiny", "small", "medium", "large", "xlarge"}


def test_model_registry_has_at_least_30_entries():
    assert len(MODEL_REGISTRY) >= 30


def test_every_model_has_non_empty_hf_id_and_valid_family():
    for name, spec in MODEL_REGISTRY.items():
        assert spec.hf_id, f"{name} has empty hf_id"
        assert spec.family in VALID_MODEL_FAMILIES, f"{name} family {spec.family}"


def test_every_model_size_tier_in_abc():
    for name, spec in MODEL_REGISTRY.items():
        assert spec.size_tier in VALID_TIERS, f"{name} tier {spec.size_tier}"


def test_list_models_by_family_returns_only_encoders():
    encoders = list_models_by_family("encoder")
    assert encoders
    for spec in encoders:
        assert spec.family == "encoder"


def test_list_models_by_tier_a_is_non_empty():
    assert list_models_by_tier("A")


def test_dataset_registry_has_exactly_23_phase_1_entries():
    assert len(DATASET_REGISTRY) == 23


def test_dataset_registry_task_type_breakdown():
    by_primary = {tt: 0 for tt in TaskType}
    for spec in DATASET_REGISTRY.values():
        by_primary[spec.task_types[0]] += 1
    assert by_primary[TaskType.CLASSIFICATION] == 4
    assert by_primary[TaskType.TOKEN_CLASSIFICATION] == 3
    assert by_primary[TaskType.PAIR_CLASSIFICATION] == 4
    assert by_primary[TaskType.STS] == 3
    assert by_primary[TaskType.RETRIEVAL] == 3
    assert by_primary[TaskType.CLUSTERING] == 3
    assert by_primary[TaskType.MULTIPLE_CHOICE_QA] == 3


def test_multi_task_datasets_have_multiple_task_types():
    multi = {name for name, spec in DATASET_REGISTRY.items() if len(spec.task_types) > 1}
    assert "scidocs" in multi


def test_every_dataset_has_valid_size_tier():
    for name, spec in DATASET_REGISTRY.items():
        assert spec.size_tier in VALID_DATASET_TIERS, f"{name} tier {spec.size_tier}"


def test_every_dataset_is_english():
    for name, spec in DATASET_REGISTRY.items():
        assert spec.language == "en", f"{name} language {spec.language}"
    assert len(list_datasets_by_language("en")) == 23


def test_no_starter_dataset_disabled_by_default():
    # conll2003 and universal_dependencies ship as old loading scripts that
    # datasets>=3 refuses to load. Until we swap them for parquet mirrors they
    # stay disabled. Every other dataset must remain enabled.
    allowed_disabled = {"conll2003", "universal_dependencies"}
    for name, spec in DATASET_REGISTRY.items():
        if name in allowed_disabled:
            continue
        assert not spec.disabled_by_default, f"{name} is disabled_by_default"


def test_retrieval_datasets_flagged_as_corpora():
    for name in ("scidocs", "arguana", "ask_ubuntu"):
        assert DATASET_REGISTRY[name].is_retrieval_corpus, name
