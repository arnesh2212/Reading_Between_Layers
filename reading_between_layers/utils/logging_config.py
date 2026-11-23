"""Project-wide logging setup.

``configure_logging`` silences the third-party loggers that flood the
console during data download and tokenizer init, and filters the user
warnings that pollute output without action. It is idempotent: calling it
twice is harmless.

Call once at the top of every Hydra entry point (after imports). The default
verbosity for our own ``__main__`` and ``reading_between_layers.*`` loggers is INFO; everything
else is WARNING or higher unless overridden via env vars.
"""

from __future__ import annotations

import logging
import os
import warnings

NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "filelock",
    "fsspec",
    "huggingface_hub",
    "datasets",
    "transformers",
    "transformers.tokenization_utils_base",
    "transformers.configuration_utils",
    "transformers.modeling_utils",
    "matplotlib",
    "PIL",
    "fontTools",
    "wandb",
)

_CONFIGURED = False


def configure_logging(level: int | str = logging.INFO) -> None:
    """Silence noisy third-party loggers and tame warnings. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        )
    else:
        root.setLevel(level)

    noisy_level = os.environ.get("NOISY_LOG_LEVEL", "WARNING").upper()
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
