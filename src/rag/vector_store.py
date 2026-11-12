"""ChromaDB-backed knowledge base using Gemini embeddings and semantic search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Settings, get_settings
from src.llm.gemini_client import GeminiClient
from src.logger import get_logger
from src.rag.chunker import DocumentChunk, build_document_chunks
from src.rag.pdf_loader import extract_pdf_text

logger = get_logger("rag.store")

DOCUMENT_CATEGORIES = (
    "Bearing Manufacturer Manual",
    "Industrial Maintenance Document",
    "Research Paper on Bearing Fault Diagnosis",
    "Vibration Analysis Reference Document",
)


class _NumpyVectorCollection:
    """Fallback persistent cosine-similarity store if Chroma native DLLs are unavailable."""

    def __init__(self, path: Path, name: str) -> None:
        self.name = name
        self._file = path / f"{name}_numpy_store.json"
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._records = data
            except Exception as exc:
                logger.warning("Could not load fallback vector store (%s), starting empty", exc)
                self._records = {}

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(self._records), encoding="utf-8")

    def count(self) -> int:
        return len(self._records)

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        for cid, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            self._records[cid] = {"document": doc, "metadata": meta, "embedding": list(emb)}
        self._save()

    def get(self, where: dict[str, Any] | None = None, include: list[str] | None = None) -> dict[str, list[Any]]:
        ids, docs, metas = [], [], []
        for cid, rec in self._records.items():
            meta = rec.get("metadata", {})
            if where and any(meta.get(k) != v for k, v in where.items()):
                continue
            ids.append(cid)
            docs.append(rec.get("document", ""))
            metas.append(meta)
        return {"ids": ids, "documents": docs, "metadatas": metas}

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None:
        if ids:
            for cid in ids:
                self._records.pop(cid, None)
        elif where:
            to_del = [
                cid
                for cid, rec in self._records.items()
                if all(rec.get("metadata", {}).get(k) == v for k, v in where.items())
            ]
            for cid in to_del:
                self._records.pop(cid, None)
        self._save()

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, list[list[Any]]]:
        if not self._records or not query_embeddings:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = np.asarray(query_embeddings[0], dtype=float)
        q_norm = float(np.linalg.norm(q)) + 1e-12
        scored: list[tuple[float, str, str, dict[str, Any]]] = []
        for cid, item in self._records.items():
            meta = item.get("metadata", {})
            if where and any(meta.get(k) != v for k, v in where.items()):
                continue
            vec = np.asarray(item["embedding"], dtype=float)
            sim = float(np.dot(q, vec) / (q_norm * (float(np.linalg.norm(vec)) + 1e-12)))
            cosine_dist = max(0.0, 1.0 - sim)
            scored.append((cosine_dist, cid, item["document"], meta))

        scored.sort(key=lambda t: t[0])
        top = scored[:n_results]
        return {
            "ids": [[t[1] for t in top]],
            "documents": [[t[2] for t in top]],
            "metadatas": [[t[3] for t in top]],
            "distances": [[t[0] for t in top]],
        }

    def clear(self) -> None:
        self._records.clear()
        if self._file.exists():
            self._file.unlink()


class KnowledgeBase:
    """ChromaDB-backed vector store of bearing manuals, maintenance docs, and research papers."""

    def __init__(self, gemini: GeminiClient, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.gemini = gemini
        self.settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._using_fallback = False
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.settings.chroma_persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.settings.chroma_collection,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Knowledge base ready via ChromaDB (%d chunks)", self.count())
        except Exception as exc:
            logger.warning("ChromaDB native backend unavailable (%s); using persistent NumPy store", exc)
            self._using_fallback = True
            self._client = None
            self._collection = _NumpyVectorCollection(
                self.settings.chroma_persist_dir, self.settings.chroma_collection
            )
            logger.info("Knowledge base ready via NumPy store (%d chunks)", self.count())

    @property
    def backend_name(self) -> str:
        return "NumPy Fallback" if self._using_fallback else "ChromaDB"

    def count(self) -> int:
        return int(self._collection.count())

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed and store pre-built :class:`DocumentChunk` objects in ChromaDB."""
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [
            {
                "document_name": c.document_name,
                "source": c.document_name,
                "page": int(c.page),
                "chunk_id": c.chunk_id,
                "chunk_index": int(c.chunk_index),
                "category": c.category,
            }
            for c in chunks
        ]
        embeddings = self.gemini.embed(docs, task_type="RETRIEVAL_DOCUMENT")
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        logger.info("Stored %d chunks in %s", len(chunks), self.backend_name)
        return len(chunks)

    def add_pdf(
        self,
        path: str | Path,
        source_name: str | None = None,
        *,
        category: str = "Vibration Analysis Reference Document",
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> int:
        """Extract, clean, chunk, embed, and store a PDF in ChromaDB. Returns chunk count."""
        path = Path(path)
        doc_name = (source_name or path.name).strip()
        pages = extract_pdf_text(path, clean=True)
        chunks = build_document_chunks(
            pages,
            document_name=doc_name,
            category=category,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not chunks:
            logger.warning("No text chunks extracted from %s", doc_name)
            return 0
        count = self.add_chunks(chunks)
        logger.info("Indexed %d chunks from %s (category=%s)", count, doc_name, category)
        return count

    def index_directory(
        self,
        directory: str | Path,
        *,
        default_category: str = "Vibration Analysis Reference Document",
    ) -> dict[str, int]:
        """Index every PDF in ``directory``."""
        results: dict[str, int] = {}
        for pdf in sorted(Path(directory).glob("*.pdf")):
            try:
                results[pdf.name] = self.add_pdf(pdf, category=default_category)
            except Exception as exc:
                logger.error("Failed to index %s: %s", pdf.name, exc)
                results[pdf.name] = -1
        return results

    def list_documents(self) -> list[dict[str, Any]]:
        """Return a summary of all documents currently stored in ChromaDB."""
        if self.count() == 0:
            return []
        raw = self._collection.get(include=["metadatas"])
        metas = raw.get("metadatas") or []
        by_doc: dict[str, dict[str, Any]] = {}
        for meta in metas:
            if not meta:
                continue
            doc_name = str(meta.get("document_name") or meta.get("source") or "unknown")
            entry = by_doc.setdefault(
                doc_name,
                {
                    "document_name": doc_name,
                    "category": str(meta.get("category", "Reference")),
                    "pages": set(),
                    "chunks": 0,
                },
            )
            if meta.get("page") is not None:
                entry["pages"].add(int(meta["page"]))
            entry["chunks"] += 1

        summaries: list[dict[str, Any]] = []
        for doc_name in sorted(by_doc):
            info = by_doc[doc_name]
            pages_sorted = sorted(info["pages"])
            summaries.append(
                {
                    "document_name": doc_name,
                    "category": info["category"],
                    "page_count": len(pages_sorted),
                    "pages": pages_sorted,
                    "chunk_count": info["chunks"],
                }
            )
        return summaries

    def delete_document(self, document_name: str) -> int:
        """Delete all chunks belonging to ``document_name``. Returns number of chunks removed."""
        if self.count() == 0:
            return 0
        existing = self._collection.get(where={"document_name": document_name}, include=["metadatas"])
        ids = existing.get("ids") or []
        if not ids:
            existing = self._collection.get(where={"source": document_name}, include=["metadatas"])
            ids = existing.get("ids") or []
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        logger.info("Deleted %d chunks for document '%s'", len(ids), document_name)
        return len(ids)

    def query(
        self,
        question: str,
        k: int = 4,
        *,
        document_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic similarity search returning the top-``k`` most relevant chunks."""
        if self.count() == 0 or not question or not question.strip():
            return []
        n_res = max(1, min(int(k), self.count()))
        q_emb = self.gemini.embed([question.strip()], task_type="RETRIEVAL_QUERY")
        where_filter = {"document_name": document_name} if document_name else None
        res = self._collection.query(query_embeddings=q_emb, n_results=n_res, where=where_filter)

        ids_list = (res.get("ids") or [[]])[0]
        docs_list = (res.get("documents") or [[]])[0]
        metas_list = (res.get("metadatas") or [[]])[0]
        dists_list = (res.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for idx, (doc, meta, dist) in enumerate(zip(docs_list, metas_list, dists_list)):
            meta = meta or {}
            cid = ids_list[idx] if idx < len(ids_list) else meta.get("chunk_id", f"chunk_{idx}")
            doc_name = str(meta.get("document_name") or meta.get("source") or "unknown")
            page_num = int(meta.get("page", 1))
            dist_f = float(dist)
            sim_score = max(0.0, min(1.0, 1.0 - dist_f))
            hits.append(
                {
                    "chunk_id": str(meta.get("chunk_id") or cid),
                    "document_name": doc_name,
                    "source": doc_name,
                    "page": page_num,
                    "category": str(meta.get("category", "Reference")),
                    "text": str(doc),
                    "distance": dist_f,
                    "similarity_score": sim_score,
                }
            )
        return hits

    @staticmethod
    def format_context(hits: list[dict[str, Any]]) -> str:
        if not hits:
            return "No reference documents available in the knowledge base."
        return "\n\n".join(
            f"[Source {i}] Document: {h.get('document_name') or h.get('source')} | "
            f"Page: {h.get('page')} | Chunk ID: {h.get('chunk_id', 'n/a')}:\n{h['text']}"
            for i, h in enumerate(hits, 1)
        )

    def reset(self) -> None:
        """Delete all stored chunks."""
        if self._using_fallback:
            self._collection.clear()
        else:
            name = self._collection.name
            self._client.delete_collection(name)
            self._collection = self._client.get_or_create_collection(
                name=name, embedding_function=None, metadata={"hnsw:space": "cosine"}
            )
        logger.info("Knowledge base reset")


_default_kb: KnowledgeBase | None = None


def get_default_knowledge_base(
    gemini: GeminiClient | None = None, settings: Settings | None = None
) -> KnowledgeBase:
    """Return or initialize the shared :class:`KnowledgeBase` instance."""
    global _default_kb
    if _default_kb is None or gemini is not None or settings is not None:
        client = gemini or GeminiClient(settings)
        _default_kb = KnowledgeBase(client, settings=settings)
    return _default_kb


def search_bearing_knowledge(
    query: str,
    top_k: int = 4,
    document_name: str | None = None,
    *,
    knowledge_base: KnowledgeBase | None = None,
) -> dict[str, Any]:
    """Retrieve the most relevant technical document chunks from ChromaDB for ``query``.

    Kept strictly separate from numerical signal processing. Returns actual
    document names, page numbers, chunk IDs, similarity scores, and text excerpts.
    """
    if not query or not str(query).strip():
        return {
            "status": "validation_error",
            "error": "Query string must not be empty.",
        }

    try:
        kb = knowledge_base or get_default_knowledge_base()
        total = kb.count()
        if total == 0:
            return {
                "status": "empty_knowledge_base",
                "query": query,
                "total_indexed_chunks": 0,
                "retrieved_chunks": [],
                "message": (
                    "The ChromaDB knowledge base currently contains 0 indexed chunks. "
                    "Do NOT fabricate document names or page numbers; inform the user that "
                    "no reference manuals or papers are indexed for this query."
                ),
            }

        k_clamped = max(1, min(int(top_k), 10))
        hits = kb.query(str(query).strip(), k=k_clamped, document_name=document_name)
        return {
            "status": "ok",
            "query": str(query).strip(),
            "total_indexed_chunks": total,
            "retrieved_chunks": [
                {
                    "chunk_id": h["chunk_id"],
                    "document_name": h["document_name"],
                    "page_number": h["page"],
                    "category": h["category"],
                    "similarity_score": round(h["similarity_score"], 4),
                    "cosine_distance": round(h["distance"], 4),
                    "text": h["text"],
                }
                for h in hits
            ],
            "citation_rules": (
                "Cite ONLY the exact document_name and page_number from retrieved_chunks "
                "(e.g., [DocumentName.pdf, Page X]). Never invent document names, page numbers, "
                "or unsupported claims. If retrieved_chunks do not cover the user's question, "
                "explicitly state that the uploaded documents do not contain that information."
            ),
        }
    except Exception as exc:
        logger.error("search_bearing_knowledge failed: %s", exc)
        return {"status": "error", "error": str(exc)}
