"""Tests for the decoder model wrapper."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from reading_between_layers.models.decoder import DecoderWrapper
from reading_between_layers.models.loader import load_model
from reading_between_layers.models.registry import get_model_spec

TINY_HF_ID = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture
def tiny_gpt2_components():
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(TINY_HF_ID)
        tokenizer = transformers.AutoTokenizer.from_pretrained(TINY_HF_ID)
    except Exception as exc:
        pytest.skip(f"tiny-gpt2 unavailable: {exc}")
    tokenizer.pad_token = None
    spec = replace(
        get_model_spec("gpt2"),
        name="tiny_random_gpt2",
        hf_id=TINY_HF_ID,
        hidden_size=model.config.hidden_size,
        num_layers=model.config.num_hidden_layers,
    )
    return spec, model, tokenizer


def test_decoder_forward_returns_one_output_per_block(tiny_gpt2_components):
    spec, model, tokenizer = tiny_gpt2_components
    wrapper = DecoderWrapper(spec, model, tokenizer)
    inputs = wrapper.tokenizer(
        ["hello there", "another input"], return_tensors="pt", padding=True
    )
    layers = wrapper.forward_with_all_layers(inputs["input_ids"], inputs["attention_mask"])
    assert len(layers) == wrapper.num_layers()
    for layer in layers:
        assert layer.hidden_states.shape[0] == 2
        assert layer.hidden_states.shape[-1] == spec.hidden_size
        assert layer.side == "decoder"


def test_decoder_sets_pad_to_eos_and_warns_once(tiny_gpt2_components, caplog):
    spec, model, tokenizer = tiny_gpt2_components
    assert tokenizer.pad_token is None
    with caplog.at_level(logging.WARNING, logger="reading_between_layers.models.decoder"):
        DecoderWrapper(spec, model, tokenizer)
    matches = [r for r in caplog.records if "pad_token" in r.getMessage()]
    assert len(matches) == 1
    assert tokenizer.pad_token == tokenizer.eos_token


def test_decoder_get_layer_module_paths_non_empty(tiny_gpt2_components):
    spec, model, tokenizer = tiny_gpt2_components
    wrapper = DecoderWrapper(spec, model, tokenizer)
    paths = wrapper.get_layer_module_paths()
    assert paths
    assert any(p in ("model.layers", "transformer.h", "gpt_neox.layers") for p in paths)


@pytest.mark.slow
def test_real_gpt2_loads_via_loader():
    spec = get_model_spec("gpt2")
    wrapper = load_model(spec, dtype="float32", quantization=None)
    assert wrapper.num_layers() == 12
    inputs = wrapper.tokenizer(["hello world"], return_tensors="pt")
    layers = wrapper.forward_with_all_layers(inputs["input_ids"], inputs["attention_mask"])
    assert len(layers) == 12
