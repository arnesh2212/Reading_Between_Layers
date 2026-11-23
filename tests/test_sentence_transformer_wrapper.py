"""Tests for the sentence-transformers model wrapper."""

from __future__ import annotations

from dataclasses import replace

import pytest

from reading_between_layers.models.registry import get_model_spec
from reading_between_layers.models.sentence_transformer import (
    POOLED_PSEUDO_LAYER_INDEX,
    SentenceTransformerWrapper,
)

SBERT_ID = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def minilm_wrapper():
    sentence_transformers = pytest.importorskip("sentence_transformers")
    transformers = pytest.importorskip("transformers")
    try:
        st_model = sentence_transformers.SentenceTransformer(SBERT_ID, device="cpu")
        tokenizer = transformers.AutoTokenizer.from_pretrained(SBERT_ID)
    except Exception as exc:
        pytest.skip(f"MiniLM SBERT unavailable: {exc}")
    underlying = st_model[0].auto_model
    spec = replace(
        get_model_spec("sbert_base"),
        name="all_minilm_l6_v2",
        hf_id=SBERT_ID,
        hidden_size=underlying.config.hidden_size,
        num_layers=underlying.config.num_hidden_layers,
    )
    return SentenceTransformerWrapper(spec, st_model, tokenizer)


def test_internal_hidden_states_have_expected_shape(minilm_wrapper):
    inputs = minilm_wrapper.tokenizer(["hi", "another"], return_tensors="pt", padding=True)
    layers = minilm_wrapper.forward_with_all_layers(
        inputs["input_ids"], inputs["attention_mask"]
    )
    assert len(layers) == minilm_wrapper.num_layers()
    for layer in layers:
        assert layer.hidden_states.shape[0] == 2
        assert layer.hidden_states.shape[-1] == minilm_wrapper.spec.hidden_size


def test_pseudo_layer_carries_pooled_baseline(minilm_wrapper):
    texts = ["a quick brown fox", "another sentence"]
    inputs = minilm_wrapper.tokenizer(texts, return_tensors="pt", padding=True)
    layers = minilm_wrapper.forward_with_layers_and_pooled(
        inputs["input_ids"], inputs["attention_mask"], texts=texts
    )
    assert layers[-1].layer_index == POOLED_PSEUDO_LAYER_INDEX
    pooled = layers[-1].hidden_states
    assert pooled.shape[0] == len(texts)
    assert pooled.shape[1] == 1
    assert pooled.shape[2] == minilm_wrapper.spec.hidden_size


def test_encode_pooled_matches_sentence_transformer_output(minilm_wrapper):
    direct = minilm_wrapper.st_model.encode(["hello"], convert_to_tensor=True)
    via_wrapper = minilm_wrapper.encode_pooled(["hello"])
    assert via_wrapper.shape == direct.shape
