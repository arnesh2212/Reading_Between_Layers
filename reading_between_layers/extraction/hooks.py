"""Forward-hook registration and removal helpers.

Used by :class:`SSMWrapper` and any future hook-based extraction. The
context-manager form guarantees handles are removed even if forward raises.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)


@contextmanager
def register_forward_hooks(
    modules: list[nn.Module],
    make_hook: Callable[[int], Callable[..., Any]],
) -> Iterator[None]:
    """Register one forward hook per module, removing all of them on exit.

    Args:
        modules: layer modules to hook, in order.
        make_hook: factory called with the module's index and returning a hook fn.
    """
    handles = []
    try:
        for i, module in enumerate(modules):
            handles.append(module.register_forward_hook(make_hook(i)))
        yield
    finally:
        for handle in handles:
            handle.remove()
