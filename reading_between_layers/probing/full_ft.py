"""Full fine-tuning probe stub. Phase 2 only.

See ``docs/spec/11_probing.md``. v1 ships frozen-feature linear and nonlinear
probes only.
"""

from __future__ import annotations

from .base import ProbeBase


class FullFTProbe(ProbeBase):
    """Reserved for the phase-2 full fine-tuning probe."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise NotImplementedError("Full fine-tuning probe is deferred to phase 2")

    def forward(self, embeddings):  # type: ignore[override]
        raise NotImplementedError("Full fine-tuning probe is deferred to phase 2")
