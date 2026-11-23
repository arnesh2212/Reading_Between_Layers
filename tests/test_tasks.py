"""Per-task feature-building and evaluation tests."""

from __future__ import annotations

import pytest
import torch

from reading_between_layers.datasets.base import DatasetSample
from reading_between_layers.tasks.classification import ClassificationTask
from reading_between_layers.tasks.clustering import ClusteringTask
from reading_between_layers.tasks.generation import GenerationTask
from reading_between_layers.tasks.multiple_choice_qa import MultipleChoiceQATask
from reading_between_layers.tasks.pair_classification import PairClassificationTask
from reading_between_layers.tasks.qa import ExtractiveQATask, GenerativeQATask
from reading_between_layers.tasks.retrieval import RetrievalTask
from reading_between_layers.tasks.sts import STSTask
from reading_between_layers.tasks.token_classification import TokenClassificationTask


def _dummy_pooled(n: int, d: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(n, d)


def test_classification_build_features_and_evaluate():
    task = ClassificationTask(num_classes=3)
    samples = [DatasetSample(inputs=["a"], label=i % 3) for i in range(6)]
    pooled = _dummy_pooled(6)
    x, y = task.build_features(samples, pooled)
    assert x.shape == (6, 8)
    assert y.shape == (6,)
    preds = torch.tensor([s.label for s in samples])
    metrics = task.evaluate(preds, y)
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0


def test_pair_classification_evaluate_returns_primary():
    task = PairClassificationTask(num_classes=3)
    samples = [DatasetSample(inputs=["a", "b"], label=i % 3) for i in range(4)]
    x, y = task.build_features(samples, _dummy_pooled(4))
    assert x.shape == (4, 8)
    metrics = task.evaluate(y, y)
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0


def test_pair_classification_rejects_negative_labels_before_cuda():
    task = PairClassificationTask(num_classes=3)
    samples = [
        DatasetSample(inputs=["a", "b"], label=0),
        DatasetSample(inputs=["c", "d"], label=-1),
    ]
    with pytest.raises(ValueError, match="labels must be in"):
        task.build_features(samples, _dummy_pooled(2))


def test_sts_build_features_returns_float_labels():
    task = STSTask()
    samples = [DatasetSample(inputs=["a", "b"], label=float(i) / 5.0) for i in range(5)]
    x, y = task.build_features(samples, _dummy_pooled(5))
    assert y.dtype == torch.float32
    metrics = task.evaluate(y, y)
    assert "spearman" in metrics


def test_clustering_evaluate_runs_kmeans_and_returns_v_measure():
    task = ClusteringTask(num_clusters=3)
    samples = [DatasetSample(inputs=["a"], label=i % 3) for i in range(30)]
    pooled = _dummy_pooled(30, d=16) + torch.tensor(
        [[i % 3] * 16 for i in range(30)], dtype=torch.float32
    )
    x, y = task.build_features(samples, pooled)
    assignments = task.cluster(x, seed=7)
    metrics = task.evaluate(assignments, y)
    assert "v_measure" in metrics
    assert 0.0 <= metrics["v_measure"] <= 1.0


def test_retrieval_score_matrix_and_evaluate():
    task = RetrievalTask()
    pooled = _dummy_pooled(8, d=8)
    samples = [DatasetSample(inputs=["q"], label=i) for i in range(8)]
    x, y = task.build_features(samples, pooled)
    assert x.shape == (8, 8)
    assert (y == torch.arange(8)).all()
    scores = task.score_matrix(x)
    metrics = task.evaluate(scores, y)
    assert "mrr_at_10" in metrics
    assert metrics["mrr_at_10"] == 1.0


def test_multiple_choice_qa_reshapes_options():
    task = MultipleChoiceQATask(num_options=4)
    samples = [DatasetSample(inputs=["q"], label=i % 4) for i in range(3)]
    pooled = _dummy_pooled(12, d=8)
    x, y = task.build_features(samples, pooled)
    assert x.shape == (3, 4, 8)
    metrics = task.evaluate(y, y)
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0


def test_multiple_choice_qa_accepts_prompt_level_embeddings():
    task = MultipleChoiceQATask(num_options=4)
    samples = [DatasetSample(inputs=["q + options"], label=i % 4) for i in range(3)]
    x, y = task.build_features(samples, _dummy_pooled(3, d=8))
    assert x.shape == (3, 8)
    assert y.tolist() == [0, 1, 2]


def test_token_classification_build_features_flattens():
    task = TokenClassificationTask(label_names=["O", "B-PER", "I-PER"])
    samples = [
        DatasetSample(inputs=["a b c"], label=[0, 1, 2]),
        DatasetSample(inputs=["d e"], label=[0, 0]),
    ]
    pooled = _dummy_pooled(5)
    x, y = task.build_features(samples, pooled)
    assert x.shape == (5, 8)
    assert y.shape == (5,)
    metrics = task.evaluate(y, y)
    assert "f1" in metrics
    assert "accuracy" in metrics


def test_qa_extractive_evaluate_returns_f1_and_em():
    task = ExtractiveQATask()
    samples = [
        DatasetSample(inputs=["q"], label="paris", extras={"answer_text": "paris"}),
        DatasetSample(inputs=["q"], label="rome", extras={"answer_text": "rome"}),
    ]
    pooled = _dummy_pooled(2)
    _, _ = task.build_features(samples, pooled)
    metrics = task.evaluate(["paris", "rome"], torch.zeros(2))
    assert "f1" in metrics
    assert metrics["f1"] == 1.0
    assert metrics["exact_match"] == 1.0


def test_qa_generative_evaluate_uses_exact_match():
    task = GenerativeQATask()
    samples = [DatasetSample(inputs=["q"], label="ans")]
    _, _ = task.build_features(samples, _dummy_pooled(1))
    metrics = task.evaluate(["ans"], torch.zeros(1))
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0


def test_generation_evaluate_handles_log_probs():
    task = GenerationTask()
    samples = [DatasetSample(inputs=["x"], label="ref") for _ in range(3)]
    _, _ = task.build_features(samples, _dummy_pooled(3))
    log_probs = torch.tensor([-1.0, -2.0, -1.5])
    metrics = task.evaluate(log_probs, torch.zeros(3))
    assert "neg_log_perplexity" in metrics
