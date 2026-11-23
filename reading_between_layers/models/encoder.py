"""Encoder-only model wrapper that pulls hidden states via ``output_hidden_states``."""

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


class EncoderWrapper(ModelWrapper):
    """Wrapper for masked-LM style encoders such as BERT, RoBERTa, DeBERTa.

    Hidden states are post-LN by construction: HuggingFace encoders return the
    output of each block after its final layer norm. Layer 0 is the embedding
    output and is skipped unless ``include_embedding`` is set on a forward call.
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
                side="single",
            )
            for i in range(start, len(hidden_states))
        ]

    def num_layers(self, side: str = "auto") -> int:
        del side
        return int(getattr(self.model.config, "num_hidden_layers", self.spec.num_layers))

    def get_layer_module_paths(self) -> list[str]:
        for path in ("encoder.layer", "transformer.layer", "layers"):
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
