"""Encoder-decoder wrapper. Supports encoder-only forward and full enc-dec forward."""

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


class EncoderDecoderWrapper(ModelWrapper):
    """Wrapper for encoder-decoder models such as T5, BART, Pegasus.

    ``forward_with_all_layers`` has two modes selected by ``which_side``:

    - ``"encoder"`` (or ``"auto"`` for non-generative tasks): runs the encoder
      only and returns its per-block hidden states.
    - ``"decoder"`` / ``"full"``: runs encoder + decoder. If ``decoder_input_ids``
      is not supplied, a single decoder-start token is used to obtain the first
      decoder hidden state. Returns decoder per-block hidden states. When
      ``decoder_attention_mask`` is supplied, it is used for downstream pooling.

    Layer indexing skips the embedding layer by default (the first hidden state
    returned by HuggingFace is the embedding output and is dropped unless
    ``include_embedding=True``).
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
        decoder_input_ids: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
    ) -> list[LayerOutput]:
        """Return per-block hidden states from the chosen side."""
        side = which_side
        if side == "auto":
            side = "encoder"
        if side == "encoder":
            return self._encoder_forward(input_ids, attention_mask, include_embedding)
        if side in ("decoder", "full"):
            return self._decoder_forward(
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                include_embedding,
            )
        raise ValueError(f"unknown side: {which_side!r}")

    def _encoder_forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        include_embedding: bool,
    ) -> list[LayerOutput]:
        with torch.inference_mode():
            outputs = self.model.encoder(
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
                side="encoder",
            )
            for i in range(start, len(hidden_states))
        ]

    def _decoder_forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        decoder_input_ids: Tensor | None,
        decoder_attention_mask: Tensor | None,
        include_embedding: bool,
    ) -> list[LayerOutput]:
        if decoder_input_ids is None:
            start_id = getattr(self.model.config, "decoder_start_token_id", None)
            if start_id is None:
                start_id = self.tokenizer.pad_token_id
            if start_id is None:
                start_id = 0
            decoder_input_ids = torch.full(
                (input_ids.size(0), 1),
                int(start_id),
                dtype=torch.long,
                device=input_ids.device,
            )
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(decoder_input_ids)
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.decoder_hidden_states
        start = 0 if include_embedding else 1
        return [
            LayerOutput(
                hidden_states=hidden_states[i],
                attention_mask=decoder_attention_mask,
                layer_index=i,
                side="decoder",
            )
            for i in range(start, len(hidden_states))
        ]

    def num_layers(self, side: str = "auto") -> int:
        cfg = self.model.config
        if side == "decoder":
            for attr in ("num_decoder_layers", "decoder_layers"):
                val = getattr(cfg, attr, None)
                if val is not None:
                    return int(val)
        for attr in ("num_layers", "encoder_layers", "num_hidden_layers"):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)
        return self.spec.num_layers

    def get_layer_module_paths(self) -> list[str]:
        paths = []
        for candidate in ("encoder.block", "model.encoder.layers", "encoder.layers"):
            if self._resolve(candidate) is not None:
                paths.append(candidate)
                break
        for candidate in ("decoder.block", "model.decoder.layers", "decoder.layers"):
            if self._resolve(candidate) is not None:
                paths.append(candidate)
                break
        return paths

    def _resolve(self, dotted: str):
        obj = self.model
        for part in dotted.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj
