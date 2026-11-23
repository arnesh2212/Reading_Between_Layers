"""State space model wrapper. Captures per-block hidden states via forward hooks."""

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

_BLOCK_LIST_PATHS: tuple[str, ...] = (
    "backbone.layers",
    "model.backbone.layers",
    "layers",
)


class SSMWrapper(ModelWrapper):
    """Wrapper for Mamba and Mamba2 state-space LMs.

    These architectures do not return per-block hidden states via the standard
    ``output_hidden_states=True`` interface in current HF integrations. We
    register forward hooks on each block in ``model.backbone.layers`` (the
    canonical path for ``state-spaces/mamba-*-hf``) and collect each block's
    output tensor. Hooks are removed in a ``finally`` block so no references
    leak even if forward raises.
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

    def _locate_blocks(self) -> tuple[nn.ModuleList, str]:
        for path in _BLOCK_LIST_PATHS:
            blocks = self._resolve(path)
            if blocks is not None:
                return blocks, path
        raise RuntimeError(
            f"could not locate SSM block list for {self.spec.name}; "
            "expected one of " + ", ".join(_BLOCK_LIST_PATHS)
        )

    def forward_with_all_layers(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        which_side: str = "auto",
        include_embedding: bool = False,
    ) -> list[LayerOutput]:
        """Run one forward pass and collect each block's output via forward hooks."""
        del which_side, include_embedding
        blocks, _ = self._locate_blocks()
        captured: list[tuple[int, Tensor]] = []
        handles: list = []

        def make_hook(idx: int):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured.append((idx, tensor.detach()))

            return hook

        try:
            for i, block in enumerate(blocks):
                handles.append(block.register_forward_hook(make_hook(i)))
            with torch.inference_mode():
                self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for handle in handles:
                handle.remove()

        captured.sort(key=lambda pair: pair[0])
        return [
            LayerOutput(
                hidden_states=tensor,
                attention_mask=attention_mask,
                layer_index=idx + 1,
                side="single",
            )
            for idx, tensor in captured
        ]

    def num_layers(self, side: str = "auto") -> int:
        del side
        cfg = self.model.config
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)
        return self.spec.num_layers

    def get_layer_module_paths(self) -> list[str]:
        _, path = self._locate_blocks()
        return [path]

    def _resolve(self, dotted: str):
        obj = self.model
        for part in dotted.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj
