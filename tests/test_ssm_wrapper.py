"""Tests for the state-space model wrapper. Marked slow: downloads mamba-130m-hf."""

from __future__ import annotations

import pytest

from reading_between_layers.models.loader import load_model
from reading_between_layers.models.registry import get_model_spec


@pytest.mark.slow
def test_mamba_130m_forward_returns_24_layers():
    spec = get_model_spec("mamba_130m")
    wrapper = load_model(spec, dtype="float32", quantization=None)
    inputs = wrapper.tokenizer(
        ["hello", "another sentence"], return_tensors="pt", padding=True
    )
    layers = wrapper.forward_with_all_layers(
        inputs["input_ids"], inputs["attention_mask"]
    )
    assert wrapper.num_layers() == 24
    assert len(layers) == 24
    seq = inputs["input_ids"].size(1)
    for layer in layers:
        assert layer.hidden_states.shape == (2, seq, 768)
