"""Tests for the extraction pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from reading_between_layers.caching.keys import EmbeddingKey
from reading_between_layers.caching.store import CacheStore
from reading_between_layers.datasets.base import DatasetSample
from reading_between_layers.extraction.batching import _bucket_by_length, build_batches
from reading_between_layers.extraction.layerwise import (
    _forward_with_extraction_side,
    extract_all_layer_embeddings,
    extract_multi_pooling_layer_embeddings,
)
from reading_between_layers.models.encoder import EncoderWrapper
from reading_between_layers.models.registry import get_model_spec
from reading_between_layers.pooling.strategies import AttentionPooling, CLSPooling, MeanPooling, SBERTPairPooling

TINY_HF_ID = "hf-internal-testing/tiny-random-bert"


def _samples(texts: list[str]) -> list[DatasetSample]:
    return [DatasetSample(inputs=[t], label=0) for t in texts]


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


def _cfg(batch_size: int = 4) -> OmegaConf:
    return OmegaConf.create(
        {
            "extraction": {
                "batch_size": batch_size,
                "max_length": 32,
                "truncation": True,
                "padding": "longest",
            }
        }
    )


class LengthOnlyTokenizer:
    def __init__(self) -> None:
        self.probe_batch_sizes: list[int] = []

    def __call__(self, texts, **kwargs):
        del kwargs
        self.probe_batch_sizes.append(len(texts))
        return {"input_ids": [[0] * len(text) for text in texts]}


class TensorTokenizer:
    def __call__(self, texts, **kwargs):
        del kwargs
        max_len = max(1, max(len(text) for text in texts))
        input_ids = []
        attention_mask = []
        for text in texts:
            length = max(1, len(text))
            input_ids.append(list(range(1, length + 1)) + [0] * (max_len - length))
            attention_mask.append([1] * length + [0] * (max_len - length))
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


class RecordingWrapper:
    def __init__(self, spec) -> None:
        self.spec = spec
        self.calls: list[dict] = []

    def forward_with_all_layers(self, input_ids, attention_mask, **kwargs):
        self.calls.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **kwargs,
            }
        )
        return []


def _keys(model_spec, num_layers: int, pooling_name: str = "cls") -> dict[int, EmbeddingKey]:
    return {
        layer: EmbeddingKey(
            model_id=model_spec.hf_id,
            dataset_id="sst2_slice",
            dataset_subset=None,
            split="train",
            layer_index=layer,
            pooling_name=pooling_name,
            dtype="float16",
            extraction_max_length=32,
            truncation=True,
        )
        for layer in range(1, num_layers + 1)
    }


def test_bucket_sort_runs_above_legacy_50k_limit():
    samples = _samples(["x" * ((i % 17) + 1) for i in range(50_001, 0, -1)])
    tokenizer = LengthOnlyTokenizer()

    sorted_samples = _bucket_by_length(samples, tokenizer, max_length=32)

    sorted_lengths = [len(sample.inputs[0]) for sample in sorted_samples]
    assert sorted_lengths == sorted(sorted_lengths)
    assert sum(tokenizer.probe_batch_sizes) == len(samples)
    assert len(tokenizer.probe_batch_sizes) > 1


def test_prefetched_batches_preserve_order():
    samples = _samples(["a", "bb", "ccc", "dddd", "eeeee"])
    batches = list(
        build_batches(
            samples,
            tokenizer=TensorTokenizer(),
            max_length=32,
            batch_size=2,
            padding="longest",
            prefetch_batches=2,
            tokenizer_workers=1,
        )
    )

    assert [[sample.inputs[0] for sample in batch["samples"]] for batch in batches] == [
        ["a", "bb"],
        ["ccc", "dddd"],
        ["eeeee"],
    ]
    assert [batch["attention_mask"].sum(dim=1).tolist() for batch in batches] == [
        [1, 2],
        [3, 4],
        [5],
    ]


def test_encoder_decoder_extraction_uses_decoder_side_and_mask():
    spec = get_model_spec("t5_v1_1_small")
    wrapper = RecordingWrapper(spec)
    input_ids = torch.tensor([[5, 6, 0], [7, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    _forward_with_extraction_side(
        wrapper,
        input_ids=input_ids,
        attention_mask=attention_mask,
        include_embedding=False,
    )

    call = wrapper.calls[0]
    assert call["which_side"] == "decoder"
    assert torch.equal(call["decoder_input_ids"], input_ids)
    assert torch.equal(call["decoder_attention_mask"], attention_mask)


def test_encoder_extraction_keeps_auto_side():
    spec = get_model_spec("bert_base_uncased")
    wrapper = RecordingWrapper(spec)
    input_ids = torch.tensor([[101, 102]])
    attention_mask = torch.tensor([[1, 1]])

    _forward_with_extraction_side(
        wrapper,
        input_ids=input_ids,
        attention_mask=attention_mask,
        include_embedding=False,
    )

    call = wrapper.calls[0]
    assert call["which_side"] == "auto"
    assert "decoder_input_ids" not in call
    assert "decoder_attention_mask" not in call


def test_end_to_end_extraction_caches_each_layer(tmp_path: Path, tiny_bert):
    cache = CacheStore(tmp_path)
    samples = _samples([f"sample number {i} with a few tokens" for i in range(50)])
    keys = _keys(tiny_bert.spec, tiny_bert.num_layers())
    cfg = _cfg(batch_size=8)

    result = extract_all_layer_embeddings(
        model=tiny_bert,
        samples=samples,
        pooling=CLSPooling(),
        config=cfg,
        cache=cache,
        embedding_keys=keys,
    )
    assert set(result.keys()) == set(keys.keys())
    for layer, tensor in result.items():
        assert tensor.shape == (50, tiny_bert.spec.hidden_size)
        assert tensor.dtype == torch.float16
        assert cache.has_embedding(keys[layer])


def test_save_only_mode_writes_cache_without_loading_results(tmp_path: Path, tiny_bert):
    cache = CacheStore(tmp_path)
    samples = _samples([f"save only sample {i}" for i in range(12)])
    keys = _keys(tiny_bert.spec, tiny_bert.num_layers())
    cfg = _cfg(batch_size=4)

    result = extract_all_layer_embeddings(
        model=tiny_bert,
        samples=samples,
        pooling=CLSPooling(),
        config=cfg,
        cache=cache,
        embedding_keys=keys,
        return_embeddings=False,
    )

    assert result == {}
    assert all(cache.has_embedding(key) for key in keys.values())

    with patch.object(
        cache,
        "load_embedding",
        side_effect=AssertionError("save-only mode should not load embeddings"),
    ):
        result = extract_all_layer_embeddings(
            model=tiny_bert,
            samples=samples,
            pooling=CLSPooling(),
            config=cfg,
            cache=cache,
            embedding_keys=keys,
            return_embeddings=False,
        )
    assert result == {}


def test_save_only_sbert_pair_caches_combined_features(tmp_path: Path, tiny_bert):
    cache = CacheStore(tmp_path)
    samples = [
        DatasetSample(inputs=[f"premise {i}", f"hypothesis {i}"], label=i % 3) for i in range(10)
    ]
    keys = _keys(
        tiny_bert.spec,
        tiny_bert.num_layers(),
        pooling_name="sbert_pair",
    )
    cfg = _cfg(batch_size=5)

    result = extract_all_layer_embeddings(
        model=tiny_bert,
        samples=samples,
        pooling=SBERTPairPooling(),
        config=cfg,
        cache=cache,
        embedding_keys=keys,
        return_embeddings=False,
    )

    assert result == {}
    first_layer = cache.load_embedding(keys[1])
    assert first_layer.shape == (len(samples), tiny_bert.spec.hidden_size * 4)
    assert first_layer.dtype == torch.float16


def test_save_only_attention_pooling_handles_inference_tensors(tmp_path: Path, tiny_bert):
    cache = CacheStore(tmp_path)
    samples = _samples([f"attention sample {i}" for i in range(8)])
    keys = _keys(
        tiny_bert.spec,
        tiny_bert.num_layers(),
        pooling_name="attention",
    )
    cfg = _cfg(batch_size=4)

    result = extract_all_layer_embeddings(
        model=tiny_bert,
        samples=samples,
        pooling=AttentionPooling(hidden_size=tiny_bert.spec.hidden_size),
        config=cfg,
        cache=cache,
        embedding_keys=keys,
        return_embeddings=False,
    )

    assert result == {}
    first_layer = cache.load_embedding(keys[1])
    assert first_layer.shape == (len(samples), tiny_bert.spec.hidden_size)
    assert first_layer.dtype == torch.float16


def test_multi_pooling_save_only_shares_sequence_forwards(tmp_path: Path, tiny_bert):
    cache = CacheStore(tmp_path)
    samples = _samples([f"multi pooling sample {i}" for i in range(9)])
    cfg = _cfg(batch_size=3)
    keys_by_pooling = {
        "cls": _keys(tiny_bert.spec, tiny_bert.num_layers(), pooling_name="cls"),
        "mean": _keys(tiny_bert.spec, tiny_bert.num_layers(), pooling_name="mean"),
    }
    original = tiny_bert.forward_with_all_layers
    call_count = 0

    def patched(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    with patch.object(tiny_bert, "forward_with_all_layers", side_effect=patched):
        result = extract_multi_pooling_layer_embeddings(
            model=tiny_bert,
            samples=samples,
            poolings={"cls": CLSPooling(), "mean": MeanPooling()},
            config=cfg,
            cache=cache,
            embedding_keys_by_pooling=keys_by_pooling,
            return_embeddings=False,
        )

    assert result == {}
    assert call_count == 3
    for keys in keys_by_pooling.values():
        assert all(cache.has_embedding(key) for key in keys.values())


def test_cuda_device_request_fails_fast_when_unavailable(tmp_path: Path, tiny_bert, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cache = CacheStore(tmp_path)
    samples = _samples(["cuda requested"])
    keys = _keys(tiny_bert.spec, tiny_bert.num_layers())
    cfg = _cfg(batch_size=1)
    cfg.device = "cuda"

    with pytest.raises(RuntimeError, match="cfg.device is cuda"):
        extract_all_layer_embeddings(
            model=tiny_bert,
            samples=samples,
            pooling=CLSPooling(),
            config=cfg,
            cache=cache,
            embedding_keys=keys,
        )


def test_rerun_hits_cache_and_does_not_call_model(tmp_path: Path, tiny_bert):
    cache = CacheStore(tmp_path)
    samples = _samples([f"another text {i}" for i in range(20)])
    keys = _keys(tiny_bert.spec, tiny_bert.num_layers())
    cfg = _cfg(batch_size=4)

    extract_all_layer_embeddings(
        model=tiny_bert,
        samples=samples,
        pooling=CLSPooling(),
        config=cfg,
        cache=cache,
        embedding_keys=keys,
    )
    with patch.object(
        tiny_bert,
        "forward_with_all_layers",
        side_effect=AssertionError("should not be called on cache hit"),
    ):
        result = extract_all_layer_embeddings(
            model=tiny_bert,
            samples=samples,
            pooling=CLSPooling(),
            config=cfg,
            cache=cache,
            embedding_keys=keys,
        )
    assert set(result.keys()) == set(keys.keys())


def test_oom_simulated_halves_batch_size(tmp_path: Path, tiny_bert):
    cache = CacheStore(tmp_path)
    samples = _samples([f"x{i}" for i in range(8)])
    keys = _keys(tiny_bert.spec, tiny_bert.num_layers())
    cfg = _cfg(batch_size=8)

    original = tiny_bert.forward_with_all_layers
    call_log: list[int] = []
    state = {"raised": False}

    def patched(input_ids, attention_mask, *args, **kwargs):
        call_log.append(int(input_ids.size(0)))
        if not state["raised"]:
            state["raised"] = True
            raise torch.cuda.OutOfMemoryError("simulated")
        return original(input_ids, attention_mask, *args, **kwargs)

    with patch.object(tiny_bert, "forward_with_all_layers", side_effect=patched):
        result = extract_all_layer_embeddings(
            model=tiny_bert,
            samples=samples,
            pooling=CLSPooling(),
            config=cfg,
            cache=cache,
            embedding_keys=keys,
        )
    assert call_log[0] == 8
    assert call_log[1] == 4
    for tensor in result.values():
        assert tensor.shape == (8, tiny_bert.spec.hidden_size)
