"""Model-aware pooling and representation-side resolution."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.registry import ModelSpec

CLS_TO_LAST_TOKEN_FAMILIES = frozenset({"encoder_decoder", "ssm"})
DECODER_REPRESENTATION_FAMILIES = frozenset({"encoder_decoder"})


def _family(model_spec: ModelSpec | None = None, family: str | None = None) -> str | None:
    if family is not None:
        return family
    if model_spec is None:
        return None
    return str(model_spec.family)


def normalize_pooling_name(
    pooling_name: str,
    *,
    model_spec: ModelSpec | None = None,
    family: str | None = None,
) -> str:
    """Return the effective pooling name for a model family.

    Encoder-decoder and SSM/Mamba models do not expose a real CLS token in this
    pipeline, so only requests for ``cls`` are redirected to ``last_token``.
    Other pooling choices are left untouched.
    """
    model_family = _family(model_spec, family)
    if pooling_name == "cls" and model_family in CLS_TO_LAST_TOKEN_FAMILIES:
        return "last_token"
    return pooling_name


def normalize_pooling_names(
    pooling_names: Iterable[str],
    *,
    model_spec: ModelSpec | None = None,
    family: str | None = None,
) -> list[str]:
    """Normalize and deduplicate pooling names while preserving order."""
    result: list[str] = []
    seen: set[str] = set()
    for name in pooling_names:
        normalized = normalize_pooling_name(name, model_spec=model_spec, family=family)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def representation_side_for_model(model_spec: ModelSpec) -> str | None:
    """Return the hidden-state side that should be cached for a model."""
    if model_spec.family in DECODER_REPRESENTATION_FAMILIES:
        return "decoder"
    return None


def forward_side_for_model(model_spec: ModelSpec) -> str:
    """Return the ``forward_with_all_layers`` side for extraction."""
    return representation_side_for_model(model_spec) or "auto"
