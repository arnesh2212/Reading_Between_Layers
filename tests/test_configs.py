"""Tests for the concrete Hydra config files under configs/."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from reading_between_layers.datasets.registry import DATASET_REGISTRY
from reading_between_layers.datasets.splits import resolve_train_split
from reading_between_layers.models.registry import MODEL_REGISTRY
from reading_between_layers.tasks.compatibility import TaskType

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")
TASK_DIR = Path(CONFIG_DIR) / "task"


@pytest.fixture(autouse=True)
def _clear_hydra():
    """Ensure each test starts with a clean Hydra global state."""
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()


def _compose(model: str, dataset: str, task: str):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(
            config_name="config",
            overrides=[f"model={model}", f"dataset={dataset}", f"task={task}"],
        )


def test_minimal_compose_yields_complete_config():
    cfg = _compose("bert_base_uncased", "sst2", "classification")
    assert cfg.model.name == "bert_base_uncased"
    assert cfg.dataset.name == "sst2"
    assert cfg.task.name == "classification"
    assert cfg.probe.type == "linear"
    assert cfg.pooling.name == "cls"


@pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY.keys()))
def test_every_model_config_loads_and_matches_registry(model_name: str):
    cfg = _compose(model_name, "sst2", "classification")
    spec = MODEL_REGISTRY[model_name]
    assert cfg.model.name == spec.name
    assert cfg.model.hf_id == spec.hf_id
    assert cfg.model.family == spec.family
    assert cfg.model.size_tier == spec.size_tier
    assert cfg.model.hidden_size == spec.hidden_size
    assert cfg.model.num_layers == spec.num_layers
    assert cfg.model.has_cls == spec.has_cls
    assert cfg.model.supports_chat_template == spec.supports_chat_template


@pytest.mark.parametrize("dataset_name", sorted(DATASET_REGISTRY.keys()))
def test_every_dataset_config_loads_and_matches_registry(dataset_name: str):
    spec = DATASET_REGISTRY[dataset_name]
    task_name = spec.task_types[0].value
    cfg = _compose("bert_base_uncased", dataset_name, task_name)
    assert cfg.dataset.name == spec.name
    assert cfg.dataset.hf_id == spec.hf_id
    assert cfg.dataset.hf_subset == spec.hf_subset
    assert cfg.dataset.language == spec.language
    assert cfg.dataset.task_type == task_name


@pytest.mark.parametrize("dataset_name", sorted(DATASET_REGISTRY.keys()))
def test_every_dataset_config_resolves_to_available_train_split(dataset_name: str):
    spec = DATASET_REGISTRY[dataset_name]
    task_name = spec.task_types[0].value
    cfg = _compose("bert_base_uncased", dataset_name, task_name)
    split = resolve_train_split(cfg.dataset, spec)
    assert split in spec.splits_available


TASK_NAMES = [t.value for t in TaskType]


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_every_task_config_has_at_least_one_allowed_family(task_name: str):
    cfg = OmegaConf.load(TASK_DIR / f"{task_name}.yaml")
    assert cfg.name == task_name
    assert len(cfg.allowed_families) >= 1
    assert cfg.metric_primary
    assert cfg.pooling_default
