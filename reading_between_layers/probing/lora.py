"""LoRA probe stub. Phase 2 only.

See ``docs/spec/11_probing.md`` for the planned design. v1 ships frozen-feature
linear and nonlinear probes only.
"""

from __future__ import annotations

from .base import ProbeBase


class LoRAProbe(ProbeBase):
    """Reserved for the phase-2 LoRA fine-tuning probe."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise NotImplementedError("LoRA probe is deferred to phase 2")

    def forward(self, embeddings):  # type: ignore[override]
        raise NotImplementedError("LoRA probe is deferred to phase 2")
