"""Tests for normalizing streamed HF dataset rows."""

from __future__ import annotations

import pytest

from reading_between_layers.datasets.materialize import iter_hf_samples, samples_from_hf_row
from reading_between_layers.datasets.registry import get_dataset_spec


def test_arc_multiple_choice_row_maps_answer_key_to_index():
    row = {
        "question": "Which force pulls objects downward?",
        "choices": {
            "text": ["magnetism", "gravity", "friction"],
            "label": ["A", "B", "C"],
        },
        "answerKey": "B",
    }
    samples = list(samples_from_hf_row(row, get_dataset_spec("arc"), "multiple_choice_qa"))
    assert len(samples) == 1
    assert samples[0].label == 1
    assert "gravity" in samples[0].inputs[0]
    assert samples[0].extras["choice_labels"] == ["A", "B", "C"]


def test_hellaswag_multiple_choice_row_uses_numeric_label():
    row = {"ctx": "The rider gets on the horse", "endings": ["a", "b", "c", "d"], "label": "2"}
    samples = list(samples_from_hf_row(row, get_dataset_spec("hellaswag"), "multiple_choice_qa"))
    assert samples[0].label == 2
    assert "2. c" in samples[0].inputs[0]


def test_piqa_multiple_choice_row_uses_solutions():
    row = {"goal": "Open the jar", "sol1": "Use a towel", "sol2": "Use a cloud", "label": 0}
    samples = list(samples_from_hf_row(row, get_dataset_spec("piqa"), "multiple_choice_qa"))
    assert samples[0].label == 0
    assert "Use a towel" in samples[0].inputs[0]


def test_multiple_choice_unlabeled_rows_keep_text_with_missing_label():
    hellaswag_row = {"ctx": "A person sits down", "endings": ["a", "b", "c", "d"], "label": ""}
    hellaswag = list(
        samples_from_hf_row(hellaswag_row, get_dataset_spec("hellaswag"), "multiple_choice_qa")
    )
    assert hellaswag[0].label is None
    assert "A person sits down" in hellaswag[0].inputs[0]

    piqa_row = {"goal": "Open the jar", "sol1": "Use a towel", "sol2": "Use a cloud", "label": -1}
    piqa = list(samples_from_hf_row(piqa_row, get_dataset_spec("piqa"), "multiple_choice_qa"))
    assert piqa[0].label is None
    assert "Use a towel" in piqa[0].inputs[0]


def test_clustering_row_expands_sentence_and_label_lists():
    row = {"sentences": ["alpha text", "beta text"], "labels": ["topic_a", "topic_b"]}
    samples = list(samples_from_hf_row(row, get_dataset_spec("arxiv_clustering_s2s"), "clustering"))
    assert [sample.inputs[0] for sample in samples] == ["alpha text", "beta text"]
    assert [sample.label for sample in samples] == ["topic_a", "topic_b"]


def test_streaming_loader_filters_snli_negative_labels(monkeypatch, capsys):
    pytest.importorskip("datasets")

    def fake_load_dataset(*args, **kwargs):
        del args, kwargs
        return [
            {"premise": "p0", "hypothesis": "h0", "label": 0},
            {"premise": "bad", "hypothesis": "missing", "label": -1},
            {"premise": "p2", "hypothesis": "h2", "label": 2},
        ]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    samples = list(
        iter_hf_samples(
            get_dataset_spec("snli"),
            split="train",
            limit=None,
            task_type="pair_classification",
            log_prefix="[test]",
        )
    )

    assert [sample.label for sample in samples] == [0, 2]
    captured = capsys.readouterr()
    assert "filtered 1 row(s) with missing/invalid supervised labels" in captured.out
