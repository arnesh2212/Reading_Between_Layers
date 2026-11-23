"""Tests for pooling strategies."""

from __future__ import annotations

import torch

from reading_between_layers.pooling import get_pooling
from reading_between_layers.pooling.selection import normalize_pooling_name, normalize_pooling_names
from reading_between_layers.pooling.strategies import (
    AttentionPooling,
    CLSPooling,
    LastTokenPooling,
    MaxPooling,
    MeanPooling,
    SBERTPairPooling,
)


def _example(batch: int = 2, seq: int = 5, hidden: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(batch, seq, hidden)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    return x, mask


def test_cls_pooling_returns_position_zero():
    x, mask = _example()
    pooled = CLSPooling().pool(x, mask)
    assert pooled.shape == (2, 4)
    assert torch.equal(pooled, x[:, 0])


def test_mean_pooling_on_all_ones_mask_matches_tensor_mean():
    torch.manual_seed(1)
    x = torch.randn(3, 6, 8)
    mask = torch.ones(3, 6)
    pooled = MeanPooling().pool(x, mask)
    assert torch.allclose(pooled, x.mean(dim=1), atol=1e-5)


def test_mean_pooling_excludes_pad_positions():
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0]])
    pooled = MeanPooling().pool(x, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]]))


def test_last_token_pooling_picks_last_non_pad_position():
    x = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    pooled = LastTokenPooling().pool(x, mask)
    assert torch.equal(pooled[0], x[0, 3])
    assert torch.equal(pooled[1], x[1, 1])


def test_max_pooling_masks_pads_with_negative_infinity():
    x = torch.tensor(
        [
            [[1.0, -1.0], [2.0, -2.0], [99.0, 99.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0]])
    pooled = MaxPooling().pool(x, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, -1.0]]))


def test_sbert_pair_pooling_output_dim_is_four_times_hidden():
    a = torch.randn(3, 7)
    b = torch.randn(3, 7)
    combined = SBERTPairPooling.combine(a, b)
    assert combined.shape == (3, 28)


def test_attention_pooling_requires_training_flag_and_returns_correct_shape():
    pool = AttentionPooling(hidden_size=4)
    assert pool.requires_training is True
    x, mask = _example()
    pooled = pool(x, mask)
    assert pooled.shape == (2, 4)


def test_attention_pooling_matches_half_precision_hidden_states():
    pool = AttentionPooling(hidden_size=4)
    x, mask = _example()

    pooled = pool(x.half(), mask)

    assert pooled.shape == (2, 4)
    assert pooled.dtype == torch.float16


def test_get_pooling_dispatches_by_name():
    assert isinstance(get_pooling("cls"), CLSPooling)
    assert isinstance(get_pooling("mean"), MeanPooling)
    assert isinstance(get_pooling("last_token"), LastTokenPooling)
    assert isinstance(get_pooling("max"), MaxPooling)
    assert isinstance(get_pooling("sbert_pair"), SBERTPairPooling)
    attn = get_pooling("attention", hidden_size=8)
    assert isinstance(attn, AttentionPooling)


def test_model_aware_pooling_rewrites_only_cls_for_encoder_decoder_and_ssm():
    assert normalize_pooling_name("cls", family="encoder_decoder") == "last_token"
    assert normalize_pooling_name("cls", family="ssm") == "last_token"
    assert normalize_pooling_name("cls", family="encoder") == "cls"
    assert normalize_pooling_name("cls", family="decoder") == "cls"
    assert normalize_pooling_name("mean", family="encoder_decoder") == "mean"


def test_model_aware_pooling_deduplicates_after_rewrite():
    assert normalize_pooling_names(["cls", "last_token"], family="ssm") == ["last_token"]
    assert normalize_pooling_names(["cls", "mean"], family="encoder_decoder") == [
        "last_token",
        "mean",
    ]
