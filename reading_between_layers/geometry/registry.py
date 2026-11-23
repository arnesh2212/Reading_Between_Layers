"""Mapping from geometry-metric name to its implementation class."""

from __future__ import annotations

from .anisotropy import AnisotropyRandomPairs
from .distance import GeodesicEuclideanRatio
from .information_theory import InformationTheoryBasic
from .intra_inter import IntraInterCosine
from .intrinsic_dimension import TwoNNIntrinsicDimension
from .isoscore import IsoScore
from .isotropy import IsotropyEigenvalueRatio, IsotropyMuViswanath
from .jer import JacobianEffectiveRank
from .lidar import LiDAR, NESum
from .participation_ratio import ParticipationRatio
from .ranks import EffectiveRank, RankMe, StableRank
from .similarity import LinearCKA
from .spectral import AlphaReQ, ConditionNumber, Top1VarianceRatio

GEOMETRY_REGISTRY: dict[str, type] = {
    "isotropy_mu_viswanath": IsotropyMuViswanath,
    "isotropy_eigenvalue_ratio": IsotropyEigenvalueRatio,
    "anisotropy_random_pairs": AnisotropyRandomPairs,
    "effective_rank": EffectiveRank,
    "stable_rank": StableRank,
    "participation_ratio": ParticipationRatio,
    "rankme": RankMe,
    "alpha_req": AlphaReQ,
    "condition_number": ConditionNumber,
    "top1_variance_ratio": Top1VarianceRatio,
    "twonn_intrinsic_dim": TwoNNIntrinsicDimension,
    "linear_cka": LinearCKA,
    "isoscore": IsoScore,
    "geodesic_euclidean_ratio": GeodesicEuclideanRatio,
    "intra_inter_cosine": IntraInterCosine,
    "lidar": LiDAR,
    "nesum": NESum,
    "jer": JacobianEffectiveRank,
    "information_theory_basic": InformationTheoryBasic,
}
