"""Abstract model wrapper interface and shared dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from torch import Tensor

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase

    from .registry import ModelSpec

Side = Literal["encoder", "decoder", "single"]


@dataclass
class LayerOutput:
    """Hidden state captured at one transformer (or SSM) layer.

    Shapes: ``hidden_states`` is ``(batch, seq, hidden)`` and ``attention_mask`` is
    ``(batch, seq)``. ``layer_index`` is 0 for the embedding layer, 1..N for
    transformer block outputs.
    """

    hidden_states: Tensor
    attention_mask: Tensor
    layer_index: int
    side: Side = "single"


class ModelWrapper(ABC):
    """Abstract wrapper around a loaded HuggingFace model.

    Subclasses know how to (a) run a single forward pass that exposes every
    layer's hidden state and (b) describe their layer module paths for hook-based
    extraction.
    """

    spec: ModelSpec
    model: nn.Module
    tokenizer: PreTrainedTokenizerBase

    @abstractmethod
    def forward_with_all_layers(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        which_side: str = "auto",
        include_embedding: bool = False,
    ) -> list[LayerOutput]:
        """Run one forward pass and return one :class:`LayerOutput` per layer."""

    @abstractmethod
    def num_layers(self, side: str = "auto") -> int:
        """Return the number of transformer (or SSM) layers, not counting embedding."""

    @abstractmethod
    def get_layer_module_paths(self) -> list[str]:
        """Return dotted attribute paths to layer modules, for hook registration."""
