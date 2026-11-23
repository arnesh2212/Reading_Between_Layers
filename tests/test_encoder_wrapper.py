"""Tests for the encoder model wrapper."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from reading_between_layers.models.encoder import EncoderWrapper
from reading_between_layers.models.loader import load_model
from reading_between_layers.models.registry import get_model_spec

TINY_HF_ID = "hf-internal-testing/tiny-random-bert"


@pytest.fixture(scope="module")
def tiny_bert():
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModel.from_pretrained(TINY_HF_ID)
        tokenizer = transformers.AutoTokenizer.from_pretrained(TINY_HF_ID)
    except Exception as exc:
        pytest.skip(f"tiny-bert unavailable: {exc}")
    spec = replace(
        get_model_spec("bert_base_uncased"),
        name="tiny_random_bert",
        hf_id=TINY_HF_ID,
        hidden_size=model.config.hidden_size,
        num_layers=model.config.num_hidden_layers,
    )
    return EncoderWrapper(spec, model, tokenizer)


def test_encoder_forward_returns_one_output_per_block(tiny_bert):
    inputs = tiny_bert.tokenizer(
        ["a b c", "d e f g", "h i", "j k l m n"],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    layers = tiny_bert.forward_with_all_layers(inputs["input_ids"], inputs["attention_mask"])
    assert len(layers) == tiny_bert.num_layers()
    seq = inputs["input_ids"].size(1)
    for layer in layers:
        assert layer.hidden_states.shape == (4, seq, tiny_bert.spec.hidden_size)
        assert layer.side == "single"
        assert layer.layer_index >= 1


def test_encoder_num_layers_matches_config(tiny_bert):
    assert tiny_bert.num_layers() == tiny_bert.model.config.num_hidden_layers


def test_encoder_include_embedding_adds_layer_zero(tiny_bert):
    inputs = tiny_bert.tokenizer(["hello world"], return_tensors="pt")
    with_embed = tiny_bert.forward_with_all_layers(
        inputs["input_ids"], inputs["attention_mask"], include_embedding=True
    )
    without = tiny_bert.forward_with_all_layers(inputs["input_ids"], inputs["attention_mask"])
    assert len(with_embed) == len(without) + 1
    assert with_embed[0].layer_index == 0


def test_encoder_get_layer_module_paths_non_empty(tiny_bert):
    paths = tiny_bert.get_layer_module_paths()
    assert paths
    assert any("layer" in p for p in paths)


@pytest.mark.slow
def test_real_bert_base_num_layers_is_twelve():
    spec = get_model_spec("bert_base_uncased")
    wrapper = load_model(spec, dtype="float32", quantization=None)
    assert wrapper.num_layers() == 12
    inputs = wrapper.tokenizer(["hello world"], return_tensors="pt")
    layers = wrapper.forward_with_all_layers(inputs["input_ids"], inputs["attention_mask"])
    assert len(layers) == 12
    assert layers[-1].hidden_states.shape[-1] == 768
    del wrapper
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
