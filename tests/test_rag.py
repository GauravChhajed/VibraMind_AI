"""Unit and live integration tests for the VibraMind AI RAG system (ChromaDB + Gemini)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from src.config import KNOWLEDGE_BASE_DIR, get_settings  # noqa: E402
from src.diagnosis import ask_with_signal_tools  # noqa: E402
from src.llm import GeminiClient, ToolExecutionContext  # noqa: E402
from src.rag import (  # noqa: E402
    KnowledgeBase,
    build_document_chunks,
    clean_pdf_text,
    extract_pdf_text,
    search_bearing_knowledge,
)
from src.signal_processing import (  # noqa: E402
    CWRU_SKF_6205_2RS,
    load_vibration_file,
)

requires_key = pytest.mark.skipif(not get_settings().has_api_key, reason="GEMINI_API_KEY not configured")


def test_clean_pdf_text_normalizes_ligatures_hyphens_and_page_numbers():
    raw = "The vibra-\ntion\ufb01lter has \ufb02at response.\n  - 12 -  \n\n\n\nSecond paragraph."
    cleaned = clean_pdf_text(raw)
    assert "vibrationfilter" in cleaned
    assert "flat response" in cleaned
    assert "- 12 -" not in cleaned
    assert "\n\n\n" not in cleaned


def test_extract_and_chunk_uploaded_pdfs():
    manual_pdf = KNOWLEDGE_BASE_DIR / "SKF_6205_Bearing_Maintenance_Manual.pdf"
    paper_pdf = KNOWLEDGE_BASE_DIR / "Randall_Antoni_Bearing_Envelope_Analysis_Paper.pdf"
    assert manual_pdf.exists() and paper_pdf.exists()

    pages_m = extract_pdf_text(manual_pdf)
    pages_p = extract_pdf_text(paper_pdf)
    assert len(pages_m) == 3
    assert len(pages_p) == 3

    chunks = build_document_chunks(
        pages_m,
        document_name=manual_pdf.name,
        category="Bearing Manufacturer Manual",
        chunk_size=500,
        overlap=100,
    )
    assert len(chunks) >= 3
    assert all(c.document_name == "SKF_6205_Bearing_Maintenance_Manual.pdf" for c in chunks)
    assert {c.page for c in chunks} == {1, 2, 3}
    assert all(c.chunk_id.startswith("SKF_6205_Bearing_Maintenance_Manual.pdf_p") for c in chunks)


@requires_key
def test_chromadb_indexing_semantic_search_and_gemini_rag_tool_calling():
    gemini = GeminiClient()
    kb = KnowledgeBase(gemini)
    kb.reset()

    manual_pdf = KNOWLEDGE_BASE_DIR / "SKF_6205_Bearing_Maintenance_Manual.pdf"
    paper_pdf = KNOWLEDGE_BASE_DIR / "Randall_Antoni_Bearing_Envelope_Analysis_Paper.pdf"

    n1 = kb.add_pdf(manual_pdf, category="Bearing Manufacturer Manual", chunk_size=600, overlap=100)
    n2 = kb.add_pdf(paper_pdf, category="Research Paper on Bearing Fault Diagnosis", chunk_size=600, overlap=100)
    assert n1 >= 3 and n2 >= 3
    assert kb.count() == n1 + n2

    docs = kb.list_documents()
    doc_names = {d["document_name"] for d in docs}
    assert doc_names == {
        "SKF_6205_Bearing_Maintenance_Manual.pdf",
        "Randall_Antoni_Bearing_Envelope_Analysis_Paper.pdf",
    }

    # 1. Direct semantic search test on the manual
    res_manual = search_bearing_knowledge(
        "What is the C3 radial internal clearance and relubrication interval for SKF 6205?",
        top_k=3,
        knowledge_base=kb,
    )
    assert res_manual["status"] == "ok"
    top_chunks = res_manual["retrieved_chunks"]
    assert len(top_chunks) >= 1
    assert any(
        c["document_name"] == "SKF_6205_Bearing_Maintenance_Manual.pdf" and c["page_number"] in (1, 2)
        for c in top_chunks
    )

    # 2. Direct semantic search test on the research paper
    res_paper = search_bearing_knowledge(
        "Why does rolling element slip smear baseband FFT harmonics and why is envelope demodulation used?",
        top_k=3,
        knowledge_base=kb,
    )
    assert res_paper["status"] == "ok"
    assert any(
        c["document_name"] == "Randall_Antoni_Bearing_Envelope_Analysis_Paper.pdf"
        for c in res_paper["retrieved_chunks"]
    )

    # 3. End-to-end Gemini Function Calling combining RAG (`search_bearing_knowledge`) + signal processing
    sig_file = Path(__file__).resolve().parent.parent / "data" / "demo_outer_race_bpfo.csv"
    loaded = load_vibration_file(sig_file, file_name=sig_file.name)
    ctx = ToolExecutionContext(
        signal=loaded.signal,
        signal_name=sig_file.name,
        fs=12_000.0,
        rpm=1797.0,
        n_balls=CWRU_SKF_6205_2RS.n_balls,
        ball_diameter=CWRU_SKF_6205_2RS.ball_diameter,
        pitch_diameter=CWRU_SKF_6205_2RS.pitch_diameter,
        contact_angle_deg=CWRU_SKF_6205_2RS.contact_angle_deg,
        knowledge_base=kb,
    )

    resp = ask_with_signal_tools(
        gemini,
        question=(
            "Use search_bearing_knowledge to look up the SKF 6205 C3 radial internal clearance "
            "and relubrication interval from the manual, and also check whether the loaded vibration "
            "signal shows outer-race fault indicators. Cite the exact document name and page number."
        ),
        tool_context=ctx,
    )
    assert resp.text
    called = {tc.name for tc in resp.tool_calls}
    assert "search_bearing_knowledge" in called
    assert "SKF_6205_Bearing_Maintenance_Manual.pdf" in resp.text

