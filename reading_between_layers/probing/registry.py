"""Mapping from probe-type name to its class."""

from __future__ import annotations

from .full_ft import FullFTProbe
from .linear import LinearProbe
from .lora import LoRAProbe
from .nonlinear import NonlinearProbe

PROBE_REGISTRY: dict[str, type] = {
    "linear": LinearProbe,
    "nonlinear": NonlinearProbe,
    "lora": LoRAProbe,
    "full_ft": FullFTProbe,
}
