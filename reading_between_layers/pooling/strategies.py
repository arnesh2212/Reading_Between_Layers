"""Concrete pooling strategies."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..utils.tensor_ops import gather_last_token, masked_max, masked_mean


class CLSPooling:
    """Return the hidden state at position 0 (the ``[CLS]`` token)."""

    name = "cls"
    requires_training = False

    def pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        del attention_mask
        return hidden_states[:, 0]


class MeanPooling:
    """Masked mean over valid token positions."""

    name = "mean"
    requires_training = False

    def pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        return masked_mean(hidden_states, attention_mask, dim=1)


class LastTokenPooling:
    """Hidden state at the last non-pad position of each row."""

    name = "last_token"
    requires_training = False

    def pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        return gather_last_token(hidden_states, attention_mask)


class MaxPooling:
    """Token-wise max with pad positions excluded."""

    name = "max"
    requires_training = False

    def pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        return masked_max(hidden_states, attention_mask, dim=1)


class AttentionPooling(nn.Module):
    """Learnable additive-attention pool. Reserved for full_ft and lora probes."""

    name = "attention"
    requires_training = True

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.query, std=0.02)

    def pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Weighted sum of token states using softmax(scores) over valid positions."""
        query = self.query.to(device=hidden_states.device, dtype=hidden_states.dtype)
        scores = hidden_states @ query
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        return self.pool(hidden_states, attention_mask)


class SBERTPairPooling:
    """SBERT-style pair feature: ``concat([a, b, |a - b|, a * b])``.

    Operates on two already-pooled sentence vectors rather than a token sequence.
    """

    name = "sbert_pair"
    requires_training = False

    def pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Not supported on a single token sequence; use :meth:`combine`."""
        del hidden_states, attention_mask
        raise NotImplementedError(
            "SBERTPairPooling operates on two pooled vectors; call combine(a, b)"
        )

    @staticmethod
    def combine(a: Tensor, b: Tensor) -> Tensor:
        """Return ``concat([a, b, |a - b|, a * b])`` along the last dimension."""
        return torch.cat([a, b, (a - b).abs(), a * b], dim=-1)
