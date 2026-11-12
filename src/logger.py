"""Logging setup shared by every VibraMind module."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from src.config import LOG_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once (console + rotating file)."""
    global _configured
    if _configured:
        return

    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger("vibramind")
    root.setLevel(getattr(logging, level_name, logging.INFO))
    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_DIR / "vibramind.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:  # read-only FS, permissions, etc.
        root.warning("File logging disabled: %s", exc)

    # Keep third-party libraries quiet unless debugging.
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use."""
    setup_logging()
    return logging.getLogger(f"vibramind.{name}")

