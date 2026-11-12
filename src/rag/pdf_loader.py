"""Extract, clean, and preprocess text from technical PDF documents using ``pypdf``."""

from __future__ import annotations

from pathlib import Path
import re
from typing import IO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.logger import get_logger

logger = get_logger("rag.pdf")

_LIGATURES = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u00a0": " ",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


class PDFExtractionError(RuntimeError):
    """Raised when a PDF cannot be opened or parsed."""


def clean_pdf_text(raw_text: str) -> str:
    """Clean and normalize raw text extracted from a technical PDF page.

    - Normalizes Unicode ligatures, smart quotes, and dashes.
    - Rejoins words split by end-of-line hyphenation (e.g. ``vibra-\\ntion`` -> ``vibration``).
    - Removes standalone page-number lines.
    - Normalizes horizontal whitespace and collapses excessive blank lines while
      preserving paragraph boundaries.
    """
    if not raw_text:
        return ""

    text = raw_text.translate(_LIGATURES)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Rejoin hyphenated line breaks between lowercase letters
    text = re.sub(r"([a-zA-Z])-\n([a-z])", r"\1\2", text)
    # Remove lines that contain only a standalone page number or "- N -"
    text = re.sub(r"(?m)^\s*(?:page\s*)?-?\s*\d+\s*-?\s*$", "", text, flags=re.IGNORECASE)
    # Collapse horizontal tabs/spaces within lines
    text = re.sub(r"[ \t]+", " ", text)
    # Trim trailing/leading spaces on each line
    text = "\n".join(line.strip() for line in text.split("\n"))
    # Collapse 3+ newlines into a single paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(source: str | Path | IO[bytes], *, clean: bool = True) -> list[dict]:
    """Return a list of ``{"page": int, "text": str}`` for pages that contain text."""
    try:
        reader = PdfReader(source)
    except (PdfReadError, OSError, ValueError) as exc:
        logger.error("Cannot open PDF: %s", exc)
        raise PDFExtractionError(f"Cannot open PDF: {exc}") from exc

    pages: list[dict] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
            text = clean_pdf_text(raw) if clean else raw.strip()
        except Exception as exc:  # malformed page - skip, don't fail the whole doc
            logger.warning("Skipping page %d: %s", i, exc)
            continue
        if text:
            pages.append({"page": i, "text": text})

    if not pages:
        logger.warning("No extractable text found (scanned PDF?)")
    return pages
