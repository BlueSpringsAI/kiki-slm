"""Structured logging with correlation IDs for Kiki SLM.

Replaces per-script basicConfig() calls with a single setup_logging()
that provides colored dev output (RichFormatter) or JSON lines for
Docker/production (JSONFormatter).

Usage:
    from kiki.utils.logging import setup_logging, set_correlation_id, log_with_data
    setup_logging()                         # dev/Colab (colored)
    setup_logging(json_output=True)         # Docker/production (JSON lines)
    setup_logging(log_file="train.log")     # dual stdout + file
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

# ---------------------------------------------------------------------------
# Correlation ID — set once per request, auto-propagates via contextvars
# ---------------------------------------------------------------------------

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current async/thread context."""
    _correlation_id.set(cid)


def get_correlation_id() -> str | None:
    """Get the current correlation ID (or None)."""
    return _correlation_id.get()


# ---------------------------------------------------------------------------
# Structured data helper
# ---------------------------------------------------------------------------

_log_data: ContextVar[dict[str, Any] | None] = ContextVar("_log_data", default=None)


def log_with_data(logger: logging.Logger, level: int, msg: str, data: dict[str, Any], **kwargs: Any) -> None:
    """Log a message with an attached structured data dict.

    The data dict is picked up by both formatters:
    - RichFormatter appends key=value pairs after the message
    - JSONFormatter includes it in the ``data`` field
    """
    _log_data.set(data)
    try:
        logger.log(level, msg, **kwargs)
    finally:
        _log_data.set(None)


# ---------------------------------------------------------------------------
# RichFormatter — colored, human-readable for dev / Colab
# ---------------------------------------------------------------------------

_LEVEL_COLORS = {
    "DEBUG": "\033[90m",    # grey
    "INFO": "\033[36m",     # cyan
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",    # red
    "CRITICAL": "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class RichFormatter(logging.Formatter):
    """``11:23:45 INFO    kiki.trainers.sft: Step 10/1500  loss=2.4312``"""

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        color = _LEVEL_COLORS.get(record.levelname, "")
        level = f"{record.levelname:<8s}"

        # Shorten logger name: kiki.trainers.sft_trainer -> kiki.trainers.sft_trainer
        name = record.name

        msg = record.getMessage()

        # Append structured data if present
        data = _log_data.get()
        if data:
            pairs = "  ".join(f"{k}={v}" for k, v in data.items())
            msg = f"{msg}  {pairs}"

        cid = _correlation_id.get()
        cid_str = f" [{cid}]" if cid else ""

        return f"{ts} {color}{level}{_RESET} {name}:{cid_str} {msg}"


# ---------------------------------------------------------------------------
# JSONFormatter — one JSON object per line for Docker / production
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Outputs one JSON object per line with ts, level, logger, msg, data."""

    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        cid = _correlation_id.get()
        if cid:
            obj["correlation_id"] = cid

        data = _log_data.get()
        if data:
            obj["data"] = data

        if record.exc_info and record.exc_info[1]:
            obj["exception"] = str(record.exc_info[1])

        return json.dumps(obj, default=str)


# ---------------------------------------------------------------------------
# setup_logging — single entry point
# ---------------------------------------------------------------------------

_NOISY_LOGGERS = (
    "urllib3",
    "httpx",
    "httpcore",
    "wandb",
    "wandb.sdk",
    "filelock",
    "fsspec",
    "datasets",
    "huggingface_hub",
    "transformers.tokenization_utils_base",
)


def setup_logging(
    *,
    level: int | str = logging.INFO,
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure structured logging for the entire process.

    Args:
        level: Root log level (default INFO).
        json_output: If True, emit JSON lines instead of colored output.
        log_file: Optional file path for dual stdout + file output.
    """
    root = logging.getLogger()

    # Clear existing handlers to avoid duplicates
    root.handlers.clear()
    root.setLevel(level)

    formatter: logging.Formatter
    if json_output:
        formatter = JSONFormatter()
    else:
        formatter = RichFormatter()

    # stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())  # always JSON for files
        root.addHandler(file_handler)

    # Silence noisy libraries
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
