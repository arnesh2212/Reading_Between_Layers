"""Sentence-Transformers wrapper exposing internal hidden states plus a pooled baseline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .base import LayerOutput
from .encoder import EncoderWrapper

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers import PreTrainedTokenizerBase

    from .registry import ModelSpec

logger = logging.getLogger(__name__)

POOLED_PSEUDO_LAYER_INDEX = -1


class SentenceTransformerWrapper(EncoderWrapper):
    """Wraps a ``sentence-transformers`` model.

    The underlying transformer's hidden states are exposed exactly as in
    :class:`EncoderWrapper`. The pooling head's output is captured separately
    by calling the full SBERT pipeline on the raw texts and returned as a
    pseudo-layer with index :data:`POOLED_PSEUDO_LAYER_INDEX`. This pseudo
    layer serves as a baseline for comparison against the internal layers.
    """

    def __init__(
        self,
        spec: ModelSpec,
        st_model: SentenceTransformer,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        underlying = st_model[0].auto_model
        super().__init__(spec, underlying, tokenizer)
        self.st_model = st_model

    def encode_pooled(self, texts: list[str]) -> Tensor:
        """Run the full SBERT pipeline and return the pooled sentence embedding."""
        return self.st_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

    def forward_with_layers_and_pooled(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        texts: list[str],
        include_embedding: bool = False,
    ) -> list[LayerOutput]:
        """Internal-layer outputs plus a pseudo-layer carrying the pooled SBERT vector."""
        layers = super().forward_with_all_layers(
            input_ids, attention_mask, include_embedding=include_embedding
        )
        pooled = self.encode_pooled(texts).to(layers[-1].hidden_states.dtype)
        pseudo_hidden = pooled.unsqueeze(1)
        pseudo_mask = torch.ones(
            pseudo_hidden.size(0),
            1,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        layers.append(
            LayerOutput(
                hidden_states=pseudo_hidden,
                attention_mask=pseudo_mask,
                layer_index=POOLED_PSEUDO_LAYER_INDEX,
                side="single",
            )
        )
        return layers
