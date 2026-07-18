"""Logging configuration shared across the platform.

We standardise on :mod:`logging` (never ``print``) so that every layer can be
observed consistently. The configuration is driven by ``settings.logging``.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

from core.config.settings import settings
from core.utils.paths import ensure_dir

_CONFIGURED = False


def _silence_benign_warnings() -> None:
    """Suppress noisy NumPy/Pandas RuntimeWarnings emitted during reductions
    (``mean``/``std``/``var``) over arrays that legitimately contain NaN values.

    NaN handling is explicit and intended throughout the platform; these
    warnings add no actionable signal and only clutter the log output.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")


def configure_logging() -> None:
    """Install a root logger based on application settings (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = settings.logging
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if cfg.to_file:
        log_dir = ensure_dir(cfg.log_abs_dir)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "platform.log"), encoding="utf-8"
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, cfg.level.upper(), logging.INFO),
        format=cfg.log_format,
        handlers=handlers,
        force=True,
    )
    _silence_benign_warnings()
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name`` (typically a module path)."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
