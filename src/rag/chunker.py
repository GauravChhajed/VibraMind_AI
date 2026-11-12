"""Meaningful text chunking with overlap and structured chunk metadata."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from src.rag.pdf_loader import clean_pdf_text


@dataclass(frozen=True)
class DocumentChunk:
    """A single document chunk with full provenance metadata for ChromaDB storage."""

    chunk_id: str
    document_name: str
    page: int
    chunk_index: int
    category: str
    text: str


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph and sentence boundaries."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size.")
    cleaned = re.sub(r"[ \t]+", " ", text).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            window = cleaned[start:end]
            min_break = int(chunk_size * 0.65)
            for sep in ("\n\n", ". ", "; ", "\n", " "):
                pos = window.rfind(sep)
                if pos >= min_break:
                    end = start + pos + len(sep)
                    break
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_document_chunks(
    pages: list[dict],
    document_name: str,
    *,
    category: str = "reference",
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Convert extracted PDF pages into structured :class:`DocumentChunk` records."""
    doc_name = document_name.strip()
    if not doc_name:
        raise ValueError("document_name must not be empty.")

    records: list[DocumentChunk] = []
    for page_entry in pages:
        page_num = int(page_entry["page"])
        cleaned = clean_pdf_text(str(page_entry.get("text", "")))
        for idx, piece in enumerate(chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)):
            digest = hashlib.sha1(f"{doc_name}|p{page_num}|c{idx}|{piece}".encode("utf-8")).hexdigest()[:12]
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", PathLikeName(doc_name))
            chunk_id = f"{slug}_p{page_num}_c{idx}_{digest}"
            records.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_name=doc_name,
                    page=page_num,
                    chunk_index=idx,
                    category=category,
                    text=piece,
                )
            )
    return records


def PathLikeName(name: str) -> str:
    return name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
