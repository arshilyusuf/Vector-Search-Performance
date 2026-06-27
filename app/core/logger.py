"""
Structured Logger.

Configures the application-wide logger with a structured JSON formatter
suitable for production log aggregation (e.g. CloudWatch, Datadog, GCP).
A human-readable format is used when DEBUG mode is active.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_LOG_FORMAT_JSON = (
    '{"time":"%(asctime)s","level":"%(levelname)s",'
    '"name":"%(name)s","message":"%(message)s"}'
)
_LOG_FORMAT_HUMAN = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_configured = False


def _configure_root(level: str = "INFO", *, debug: bool = False) -> None:
    """Set up the root logger exactly once.

    Args:
        level: Logging level string (e.g. 'INFO', 'DEBUG').
        debug: If True, use a human-readable format instead of JSON.
    """
    global _configured
    if _configured:
        return

    fmt = _LOG_FORMAT_HUMAN if debug else _LOG_FORMAT_JSON
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)

    # Silence overly verbose third-party loggers
    for noisy in ("uvicorn.access", "httpx", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger, initialising the root logger if necessary.

    This function is the single entry point for obtaining loggers across
    the application. It reads the LOG_LEVEL and DEBUG settings on first
    call.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    # Lazy import to avoid circular deps with config
    from app.core.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    _configure_root(level=settings.LOG_LEVEL, debug=settings.DEBUG)
    return logging.getLogger(name or "app")
