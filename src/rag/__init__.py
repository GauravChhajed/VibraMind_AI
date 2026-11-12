"""Retrieval-Augmented Generation: PDF extraction, cleaning, chunking, and ChromaDB search."""

from src.rag.chunker import DocumentChunk, build_document_chunks, chunk_text
from src.rag.pdf_loader import PDFExtractionError, clean_pdf_text, extract_pdf_text
from src.rag.vector_store import (
    DOCUMENT_CATEGORIES,
    KnowledgeBase,
    get_default_knowledge_base,
    search_bearing_knowledge,
)

__all__ = [
    "DOCUMENT_CATEGORIES",
    "DocumentChunk",
    "KnowledgeBase",
    "PDFExtractionError",
    "build_document_chunks",
    "clean_pdf_text",
    "chunk_text",
    "extract_pdf_text",
    "get_default_knowledge_base",
    "search_bearing_knowledge",
]
