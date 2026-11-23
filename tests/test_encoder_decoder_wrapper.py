"""Tests for the encoder-decoder model wrapper."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from reading_between_layers.models.encoder_decoder import EncoderDecoderWrapper
from reading_between_layers.models.registry import get_model_spec

TINY_HF_ID = "hf-internal-testing/tiny-random-T5Model"


@pytest.fixture(scope="module")
def tiny_t5():
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModel.from_pretrained(TINY_HF_ID)
        tokenizer = transformers.AutoTokenizer.from_pretrained(TINY_HF_ID)
    except Exception as exc:
        pytest.skip(f"tiny T5 unavailable: {exc}")
    spec = replace(
        get_model_spec("t5_v1_1_small"),
        name="tiny_random_t5",
        hf_id=TINY_HF_ID,
        hidden_size=model.config.d_model,
        num_layers=model.config.num_layers,
    )
    return EncoderDecoderWrapper(spec, model, tokenizer)


def test_encoder_only_forward_returns_block_outputs(tiny_t5):
    inputs = tiny_t5.tokenizer(
        ["translate this", "another sentence"],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    layers = tiny_t5.forward_with_all_layers(
        inputs["input_ids"], inputs["attention_mask"], which_side="encoder"
    )
    assert layers
    for layer in layers:
        assert layer.side == "encoder"
        assert layer.hidden_states.shape[0] == 2
        assert layer.hidden_states.shape[-1] == tiny_t5.model.config.d_model
        assert layer.layer_index >= 1


def test_auto_side_defaults_to_encoder(tiny_t5):
    inputs = tiny_t5.tokenizer(["hello"], return_tensors="pt")
    layers = tiny_t5.forward_with_all_layers(inputs["input_ids"], inputs["attention_mask"])
    assert all(layer.side == "encoder" for layer in layers)


def test_decoder_side_forward_returns_decoder_hidden_states(tiny_t5):
    inputs = tiny_t5.tokenizer(["hello"], return_tensors="pt")
    layers = tiny_t5.forward_with_all_layers(
        inputs["input_ids"], inputs["attention_mask"], which_side="decoder"
    )
    assert layers
    for layer in layers:
        assert layer.side == "decoder"
        assert layer.hidden_states.shape[1] == 1  # single decoder-start token
        assert layer.hidden_states.shape[-1] == tiny_t5.model.config.d_model


def test_full_side_accepts_explicit_decoder_input_ids(tiny_t5):
    inputs = tiny_t5.tokenizer(["hello"], return_tensors="pt")
    decoder_ids = torch.zeros((1, 3), dtype=torch.long)
    layers = tiny_t5.forward_with_all_layers(
        inputs["input_ids"],
        inputs["attention_mask"],
        which_side="full",
        decoder_input_ids=decoder_ids,
    )
    for layer in layers:
        assert layer.hidden_states.shape[1] == 3


def test_decoder_side_uses_explicit_decoder_attention_mask(tiny_t5):
    inputs = tiny_t5.tokenizer(["hello", "a longer input"], return_tensors="pt", padding=True)
    decoder_ids = inputs["input_ids"]
    decoder_mask = inputs["attention_mask"]

    layers = tiny_t5.forward_with_all_layers(
        inputs["input_ids"],
        inputs["attention_mask"],
        which_side="decoder",
        decoder_input_ids=decoder_ids,
        decoder_attention_mask=decoder_mask,
    )

    assert layers
    for layer in layers:
        assert layer.side == "decoder"
        assert layer.hidden_states.shape[1] == decoder_ids.shape[1]
        assert torch.equal(layer.attention_mask, decoder_mask)


def test_get_layer_module_paths_returns_both_sides(tiny_t5):
    paths = tiny_t5.get_layer_module_paths()
    assert any("encoder" in p for p in paths)
    assert any("decoder" in p for p in paths)
