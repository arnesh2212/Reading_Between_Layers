"""Causal decoder wrapper. Handles pad-token policy and optional chat templating."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .base import LayerOutput, ModelWrapper

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase

    from .registry import ModelSpec

logger = logging.getLogger(__name__)


class DecoderWrapper(ModelWrapper):
    """Wrapper for causal language models such as GPT-2, Llama, Gemma, Qwen.

    Hidden states are post-block residuals. GPT-2 is pre-LN architecturally,
    so the final block output is also run through ``ln_f`` by HuggingFace and
    appears as the last entry of ``outputs.hidden_states``. Intermediate
    layer outputs are raw block residuals.

    If the tokenizer lacks a pad token, ``pad_token`` is set to ``eos_token``
    and a warning is logged once.
    """

    def __init__(
        self,
        spec: ModelSpec,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.spec = spec
        self.model = model
        self.tokenizer = tokenizer
        if tokenizer.pad_token is None:
            logger.warning(
                "tokenizer for %s has no pad_token; setting pad_token = eos_token",
                spec.name,
            )
            tokenizer.pad_token = tokenizer.eos_token
            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = tokenizer.pad_token_id

    def forward_with_all_layers(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        which_side: str = "auto",
        include_embedding: bool = False,
    ) -> list[LayerOutput]:
        """Return one :class:`LayerOutput` per hidden state, indexed 1..N by default."""
        del which_side
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        start = 0 if include_embedding else 1
        return [
            LayerOutput(
                hidden_states=hidden_states[i],
                attention_mask=attention_mask,
                layer_index=i,
                side="decoder",
            )
            for i in range(start, len(hidden_states))
        ]

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Format ``messages`` via the tokenizer's chat template.

        Only valid when the model's spec advertises ``supports_chat_template``.
        """
        if not self.spec.supports_chat_template:
            raise ValueError(f"{self.spec.name} does not advertise a chat template")
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def num_layers(self, side: str = "auto") -> int:
        del side
        cfg = self.model.config
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)
        return self.spec.num_layers

    def get_layer_module_paths(self) -> list[str]:
        for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
            if self._resolve(path) is not None:
                return [path]
        return []

    def _resolve(self, dotted: str):
        obj = self.model
        for part in dotted.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj
