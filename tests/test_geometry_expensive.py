"""Tests for the expensive geometry metrics added in prompt 15."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from reading_between_layers.geometry.information_theory import (
    InformationTheoryBasic,
    _kozachenko_leonenko_entropy,
)
from reading_between_layers.geometry.jer import JacobianEffectiveRank
from reading_between_layers.geometry.lidar import LiDAR, NESum
from reading_between_layers.geometry.registry import GEOMETRY_REGISTRY


def test_nesum_identity_equals_n():
    n = 32
    out = NESum().compute(torch.eye(n))
    assert abs(out["nesum"] - n) < 1e-3


def test_kozachenko_leonenko_close_to_analytic_gaussian():
    torch.manual_seed(0)
    d = 10
    n = 5_000
    samples = torch.randn(n, d).numpy()
    estimate = _kozachenko_leonenko_entropy(samples, k=3)
    analytic = 0.5 * d * math.log(2 * math.pi * math.e)
    assert abs(estimate - analytic) < 2.0


def test_information_theory_basic_returns_finite_entropy():
    torch.manual_seed(0)
    out = InformationTheoryBasic(k=3, pca_dim=8).compute(torch.randn(200, 16))
    assert "representation_entropy" in out
    assert math.isfinite(out["representation_entropy"])


def test_information_theory_class_conditional_with_labels():
    torch.manual_seed(0)
    x = torch.randn(120, 8)
    labels = torch.tensor([i % 3 for i in range(120)])
    out = InformationTheoryBasic(k=3, pca_dim=8).compute(x, labels=labels)
    assert math.isfinite(out["class_conditional_entropy"])


def test_lidar_without_second_view_falls_back_to_nesum():
    torch.manual_seed(0)
    x = torch.randn(200, 16)
    out_lidar = LiDAR().compute(x)
    out_nesum = NESum().compute(x)
    assert abs(out_lidar["lidar"] - out_nesum["nesum"]) < 1e-3


def test_lidar_two_views_returns_finite_scalar():
    torch.manual_seed(0)
    view_a = torch.randn(64, 16)
    view_b = view_a + 0.1 * torch.randn_like(view_a)
    out = LiDAR().compute(view_a, labels=view_b)
    assert math.isfinite(out["lidar"])


def test_jer_returns_positive_scalar_on_tiny_network():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 12), nn.Tanh(), nn.Linear(12, 6))
    x = torch.randn(20, 8)
    out = JacobianEffectiveRank(sample_cap=10).compute(x, model=model)
    assert math.isfinite(out["jer"])
    assert out["jer"] > 0.0


def test_expensive_metrics_registered():
    for name in ("lidar", "nesum", "jer", "information_theory_basic"):
        assert name in GEOMETRY_REGISTRY


@pytest.mark.slow
def test_lidar_with_dropout_views_finite():
    torch.manual_seed(0)
    encoder = nn.Sequential(nn.Linear(32, 16), nn.Dropout(p=0.2), nn.ReLU(), nn.Linear(16, 16))
    encoder.train()
    x = torch.randn(48, 32)
    view_a = encoder(x).detach()
    view_b = encoder(x).detach()
    out = LiDAR().compute(view_a, labels=view_b)
    assert math.isfinite(out["lidar"])
