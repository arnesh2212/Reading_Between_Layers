"""Family-aware model loader that respects size tiers and quantization."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from ..exceptions import TierSkipError
from .decoder import DecoderWrapper
from .encoder import EncoderWrapper
from .encoder_decoder import EncoderDecoderWrapper
from .sentence_transformer import SentenceTransformerWrapper
from .ssm import SSMWrapper

if TYPE_CHECKING:
    from .base import ModelWrapper
    from .registry import ModelSpec

logger = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
_TIER_C_OVERRIDE_ENV = "RBL_ALLOW_TIER_C"


def load_model(
    spec: ModelSpec,
    dtype: str = "float16",
    quantization: str | None = None,
) -> ModelWrapper:
    """Load and wrap a model from the HuggingFace Hub by family.

    Args:
        spec: registry entry for the model.
        dtype: ``float16``, ``bfloat16``, or ``float32``.
        quantization: ``int8`` or ``int4`` for tier-B models; ``None`` otherwise.

    Returns:
        A family-appropriate :class:`ModelWrapper`.

    Raises:
        TierSkipError: if ``spec`` is tier C and ``RBL_ALLOW_TIER_C=1`` is not set.
        NotImplementedError: if ``spec.family`` is not recognized.
    """
    if spec.size_tier == "C" and os.environ.get(_TIER_C_OVERRIDE_ENV) != "1":
        raise TierSkipError(
            f"{spec.name} is tier C; set {_TIER_C_OVERRIDE_ENV}=1 to load"
        )

    if dtype not in _DTYPE_MAP:
        raise ValueError(f"unsupported dtype: {dtype}")
    torch_dtype = _DTYPE_MAP[dtype]

    trust_remote_code = bool(getattr(spec, "trust_remote_code", False))
    kwargs: dict = {"torch_dtype": torch_dtype}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if quantization in ("int8", "int4"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=(quantization == "int8"),
            load_in_4bit=(quantization == "int4"),
        )

    if spec.family == "sentence_transformer":
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(spec.hf_id, device="cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id, trust_remote_code=trust_remote_code
        )
        return SentenceTransformerWrapper(spec, st_model, tokenizer)

    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=trust_remote_code
    )

    if spec.family == "encoder":
        model = AutoModel.from_pretrained(spec.hf_id, **kwargs)
        return EncoderWrapper(spec, model, tokenizer)
    if spec.family == "decoder":
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
        return DecoderWrapper(spec, model, tokenizer)
    if spec.family == "encoder_decoder":
        model = AutoModel.from_pretrained(spec.hf_id, **kwargs)
        return EncoderDecoderWrapper(spec, model, tokenizer)
    if spec.family == "ssm":
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
        return SSMWrapper(spec, model, tokenizer)
    raise NotImplementedError(
        f"loader does not yet handle family {spec.family!r} for {spec.name}"
    )
