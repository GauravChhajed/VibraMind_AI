"""Centralised application configuration.

Settings are read from environment variables, optionally loaded from a
``.env`` file in the project root. Secrets (the Gemini API key) are never
hard-coded, logged or exposed through ``repr``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
LOG_DIR = PROJECT_ROOT / "logs"

# Load .env once at import time. Existing environment variables take precedence.
load_dotenv(PROJECT_ROOT / ".env", override=False)

_PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here"}


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # repr=False keeps the key out of logs, tracebacks and Streamlit output.
    gemini_api_key: str | None = field(default=None, repr=False)
    gemini_model: str = "gemini-3.8-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    log_level: str = "INFO"
    chroma_persist_dir: Path = DATA_DIR / "chroma_db"
    chroma_collection: str = "vibramind_knowledge"

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_api_key) and self.gemini_api_key not in _PLACEHOLDER_KEYS

    def require_api_key(self) -> str:
        """Return the API key or raise a helpful error if it's not configured."""
        if not self.has_api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. Copy `.env.example` to `.env` and add "
                "your key from https://aistudio.google.com/apikey"
            )
        return self.gemini_api_key  # type: ignore[return-value]


def _resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_settings() -> Settings:
    """Build settings from the current environment."""
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip() or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip(),
        gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip(),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        chroma_persist_dir=_resolve_path(os.getenv("CHROMA_PERSIST_DIR"), DATA_DIR / "chroma_db"),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "vibramind_knowledge").strip(),
    )

