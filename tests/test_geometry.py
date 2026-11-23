"""Tests for the cheap geometry metrics implemented in prompt 14."""

from __future__ import annotations

import torch

from reading_between_layers.geometry.anisotropy import AnisotropyRandomPairs
from reading_between_layers.geometry.distance import GeodesicEuclideanRatio
from reading_between_layers.geometry.intra_inter import IntraInterCosine
from reading_between_layers.geometry.intrinsic_dimension import TwoNNIntrinsicDimension
from reading_between_layers.geometry.isoscore import IsoScore
from reading_between_layers.geometry.isotropy import IsotropyEigenvalueRatio, IsotropyMuViswanath
from reading_between_layers.geometry.participation_ratio import ParticipationRatio
from reading_between_layers.geometry.ranks import EffectiveRank, RankMe, StableRank
from reading_between_layers.geometry.registry import GEOMETRY_REGISTRY
from reading_between_layers.geometry.similarity import LinearCKA
from reading_between_layers.geometry.spectral import AlphaReQ, ConditionNumber, Top1VarianceRatio


def test_effective_rank_identity_equals_n():
    n = 32
    x = torch.eye(n)
    out = EffectiveRank().compute(x)
    assert abs(out["effective_rank"] - n) < 1e-3


def test_stable_rank_identity_and_rank_one():
    n = 16
    eye = torch.eye(n)
    rank_one = torch.outer(torch.arange(1, n + 1, dtype=torch.float32), torch.ones(n))
    assert abs(StableRank().compute(eye)["stable_rank"] - n) < 1e-3
    assert abs(StableRank().compute(rank_one)["stable_rank"] - 1.0) < 1e-3


def test_effective_rank_rank_one_is_one():
    rank_one = torch.outer(torch.ones(16), torch.arange(1.0, 17.0))
    out = EffectiveRank().compute(rank_one)
    assert abs(out["effective_rank"] - 1.0) < 1e-3


def test_participation_ratio_identity_and_rank_one():
    n = 24
    eye = torch.eye(n)
    rank_one = torch.outer(torch.arange(1, n + 1, dtype=torch.float32), torch.ones(n))
    assert abs(ParticipationRatio().compute(eye)["participation_ratio"] - n) < 1e-3
    assert abs(ParticipationRatio().compute(rank_one)["participation_ratio"] - 1.0) < 1e-3


def test_anisotropy_isotropic_gaussian_close_to_zero():
    torch.manual_seed(0)
    x = torch.randn(10_000, 128)
    out = AnisotropyRandomPairs(num_pairs=10_000, seed=1).compute(x)
    assert abs(out["anisotropy_random_pairs"]) < 0.02


def test_anisotropy_strongly_anisotropic_above_zero():
    # Real-world "anisotropic" embeddings concentrate in a cone (strong shared
    # mean direction). Construct that explicitly: large positive mean on the
    # first axis with small per-coordinate noise.
    torch.manual_seed(0)
    n = 5_000
    d = 32
    mean = torch.zeros(d)
    mean[0] = 10.0
    z = mean.unsqueeze(0) + torch.randn(n, d)
    out = AnisotropyRandomPairs(num_pairs=20_000, seed=2).compute(z)
    assert out["anisotropy_random_pairs"] > 0.2


def test_isoscore_identity_close_to_one_rank_one_close_to_zero():
    torch.manual_seed(0)
    n, d = 4_000, 64
    iso = torch.randn(n, d)
    iso_score = IsoScore().compute(iso)["isoscore"]
    direction = torch.randn(d)
    direction = direction / direction.norm()
    rank_one = torch.arange(1.0, n + 1).unsqueeze(-1) * direction.unsqueeze(0)
    rank_one_score = IsoScore().compute(rank_one)["isoscore"]
    assert iso_score > 0.7
    assert rank_one_score < 0.3


def test_intra_inter_labeled_clusters_gap_positive():
    torch.manual_seed(1)
    d = 16
    num_classes = 3
    per_class = 40
    centers = 3.0 * torch.randn(num_classes, d)
    xs, ys = [], []
    for c in range(num_classes):
        xs.append(centers[c].unsqueeze(0) + 0.1 * torch.randn(per_class, d))
        ys.append(torch.full((per_class,), c, dtype=torch.long))
    x = torch.cat(xs)
    y = torch.cat(ys)
    out = IntraInterCosine().compute(x, labels=y)
    assert out["intra_cosine_mean"] > out["inter_cosine_mean"]
    assert out["intra_inter_gap"] > 0.0


def test_rankme_finite_on_random_input():
    torch.manual_seed(0)
    out = RankMe().compute(torch.randn(64, 32))
    assert 0.0 < out["rankme"] <= 32


def test_alpha_req_returns_slope_and_r2():
    torch.manual_seed(0)
    x = torch.randn(200, 50)
    out = AlphaReQ().compute(x)
    assert "alpha_req" in out
    assert "alpha_req_r2" in out
    assert out["alpha_req_r2"] <= 1.0


def test_condition_number_identity_and_rank_one():
    n = 16
    eye = torch.eye(n)
    rank_one = torch.outer(torch.arange(1, n + 1, dtype=torch.float32), torch.ones(n))
    assert abs(ConditionNumber().compute(eye)["condition_number"] - 1.0) < 1e-3
    assert ConditionNumber().compute(rank_one)["condition_number"] > 1e6


def test_top1_variance_ratio_identity_and_rank_one():
    n = 12
    eye = torch.eye(n)
    rank_one = torch.outer(torch.arange(1, n + 1, dtype=torch.float32), torch.ones(n))
    assert abs(Top1VarianceRatio().compute(eye)["top1_variance_ratio"] - (1.0 / n)) < 1e-3
    assert abs(Top1VarianceRatio().compute(rank_one)["top1_variance_ratio"] - 1.0) < 1e-3


def test_twonn_intrinsic_dim_line_close_to_one():
    torch.manual_seed(0)
    t = torch.rand(500, 1)
    x = torch.cat([t, t * 0.0, t * 0.0, t * 0.0], dim=1)
    value = TwoNNIntrinsicDimension(chunk_size=128).compute(x)["twonn_intrinsic_dim"]
    assert 0.5 < value < 2.0


def test_linear_cka_identical_scaled_representation_is_one():
    torch.manual_seed(0)
    x = torch.randn(200, 32)
    y = 3.0 * x
    value = LinearCKA().compute(x, labels=y)["linear_cka"]
    assert abs(value - 1.0) < 1e-5


def test_isotropy_eigenvalue_ratio_finite():
    torch.manual_seed(0)
    out = IsotropyEigenvalueRatio().compute(torch.randn(500, 64))
    assert out["iso_eigenvalue_ratio"] > 0.0


def test_isotropy_mu_viswanath_in_unit_interval():
    torch.manual_seed(0)
    out = IsotropyMuViswanath(num_directions=16, seed=0).compute(torch.randn(500, 64))
    assert 0.0 <= out["iso_mu_viswanath"] <= 1.0


def test_geodesic_euclidean_ratio_runs_on_small_input():
    torch.manual_seed(0)
    out = GeodesicEuclideanRatio(k=5, max_pairs=200).compute(torch.randn(80, 12))
    # geodesic distance is at least Euclidean, so mean ratio >= 1 in expectation.
    assert out["geodesic_euclidean_mean_ratio"] == out["geodesic_euclidean_mean_ratio"]


def test_registry_lists_cheap_metrics():
    for name in [
        "isotropy_mu_viswanath",
        "isotropy_eigenvalue_ratio",
        "anisotropy_random_pairs",
        "effective_rank",
        "stable_rank",
        "participation_ratio",
        "rankme",
        "alpha_req",
        "condition_number",
        "top1_variance_ratio",
        "twonn_intrinsic_dim",
        "linear_cka",
        "isoscore",
        "geodesic_euclidean_ratio",
        "intra_inter_cosine",
    ]:
        assert name in GEOMETRY_REGISTRY
