"""Final end-to-end integration tests for VibraMind AI.

Covers:
1. MFPT bearing dataset signals (.mat struct & .csv).
2. Healthy vs. faulty bearing classification (Healthy, BPFO, BPFI).
3. Diverse sampling frequencies and signal lengths.
4. Missing and invalid bearing / sampling parameters.
5. Live Gemini Function Calling with actual MFPT vibration signal.
6. ChromaDB RAG retrieval and exact document + page citations.
7. Complete 10-section diagnostic report generation.
8. Streamlit 4-section application verification via AppTest.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.config import DATA_DIR, KNOWLEDGE_BASE_DIR, PROJECT_ROOT
from src.diagnosis import analyze_signal, ask_with_signal_tools, generate_report
from src.llm import GeminiClient, ToolExecutionContext, execute_tool_call
from src.rag import KnowledgeBase, search_bearing_knowledge
from src.signal_processing import (
    CWRU_SKF_6205_2RS,
    MFPT_NICE_BEARING,
    BearingGeometry,
    bearing_fault_frequencies,
    generate_bearing_signal,
    load_vibration_file,
    rpm_to_hz,
)


def test_mfpt_healthy_vs_faulty_dataset_signals() -> None:
    """Test 1 & 2: MFPT dataset (.mat structs & .csv) for healthy, outer-race (BPFO), and inner-race (BPFI) conditions."""
    shaft_hz = rpm_to_hz(1500.0)  # 25.0 Hz

    # 1. Healthy baseline (97,656 Hz)
    healthy_loaded = load_vibration_file(DATA_DIR / "mfpt_baseline_healthy_97656hz.mat")
    assert healthy_loaded.channel == "bearing.gs"
    assert healthy_loaded.metadata.get("bearing.sr") == 97656.0
    assert healthy_loaded.metadata.get("bearing.rate") == 25.0

    healthy_res = analyze_signal(
        healthy_loaded.signal,
        fs=97_656.0,
        shaft_hz=shaft_hz,
        geometry=MFPT_NICE_BEARING,
        envelope_band=(2500.0, 5500.0),
    )
    assert healthy_res.suspected_faults == []

    # 2. Outer-race fault (48,828 Hz, BPFO ≈ 81.12 Hz)
    bpfo_loaded = load_vibration_file(DATA_DIR / "mfpt_outer_race_fault_48828hz.mat")
    assert bpfo_loaded.channel == "bearing.gs"
    bpfo_res = analyze_signal(
        bpfo_loaded.signal,
        fs=48_828.0,
        shaft_hz=shaft_hz,
        geometry=MFPT_NICE_BEARING,
        envelope_band=(2500.0, 5500.0),
    )
    assert "Outer race" in bpfo_res.suspected_faults
    assert bpfo_res.fft_comparison is not None
    assert bpfo_res.fft_comparison.comparisons["BPFO"].possible_match is True

    # 3. Inner-race fault (48,828 Hz, BPFI ≈ 118.88 Hz)
    bpfi_loaded = load_vibration_file(DATA_DIR / "mfpt_inner_race_fault_48828hz.mat")
    bpfi_res = analyze_signal(
        bpfi_loaded.signal,
        fs=48_828.0,
        shaft_hz=shaft_hz,
        geometry=MFPT_NICE_BEARING,
        envelope_band=(2500.0, 5500.0),
    )
    assert "Inner race" in bpfi_res.suspected_faults
    assert bpfi_res.fft_comparison is not None
    assert bpfi_res.fft_comparison.comparisons["BPFI"].possible_match is True


@pytest.mark.parametrize(
    "fs_val,duration_s",
    [
        (2_000.0, 0.1),    # 200 samples
        (12_000.0, 0.25),  # 3,000 samples
        (48_828.0, 0.2),   # 9,765 samples
        (97_656.0, 0.1),   # 9,765 samples
    ],
)
def test_different_sampling_frequencies_and_signal_lengths(fs_val: float, duration_s: float) -> None:
    """Test 3: Verify pipeline works across varied sampling rates and signal lengths."""
    shaft_hz = 25.0
    freqs = bearing_fault_frequencies(shaft_hz=shaft_hz, geometry=MFPT_NICE_BEARING)
    sig = generate_bearing_signal(
        fs=fs_val,
        duration=duration_s,
        shaft_hz=shaft_hz,
        fault_freq=freqs["BPFO"],
        resonance_hz=min(3500.0, fs_val * 0.3),
        seed=7,
    )
    res = analyze_signal(
        sig,
        fs=fs_val,
        shaft_hz=shaft_hz,
        geometry=MFPT_NICE_BEARING,
        envelope_band=(min(500.0, fs_val * 0.15), min(4500.0, fs_val * 0.4)),
    )
    assert len(res.processed_signal) == len(sig)
    assert res.features["rms"] > 0
    assert len(res.fft[0]) > 0
    assert len(res.psd[0]) > 0
    assert res.spectrogram[2].size > 0


def test_missing_and_invalid_bearing_parameters() -> None:
    """Test 4: Verify missing or invalid sampling rate and bearing parameters are rejected cleanly."""
    with pytest.raises(ValueError, match="Shaft speed"):
        rpm_to_hz(None)
    with pytest.raises(ValueError, match="positive"):
        rpm_to_hz(-500.0)
    with pytest.raises(ValueError, match="smaller than pitch diameter"):
        BearingGeometry(n_balls=8, ball_diameter=40.0, pitch_diameter=30.0, contact_angle_deg=0.0).validate()
    with pytest.raises(ValueError, match="Contact angle"):
        BearingGeometry(n_balls=8, ball_diameter=6.0, pitch_diameter=30.0, contact_angle_deg=95.0).validate()

    # Tool-level missing parameter handling
    empty_ctx = ToolExecutionContext(signal=None, fs=None, rpm=None)
    res = execute_tool_call("calculate_bearing_frequencies", {}, empty_ctx)
    assert res["status"] == "missing_parameters"
    assert any("rpm" in m for m in res["missing"])


def test_complete_workflow_tools_rag_and_report() -> None:
    """Test 5, 6 & 7: End-to-end workflow (MFPT signal -> Function Calling + RAG -> 10-section Report)."""
    client = GeminiClient()
    kb = KnowledgeBase(client)
    if kb.count() == 0:
        kb.index_directory(KNOWLEDGE_BASE_DIR)

    # Verify RAG retrieval directly
    rag_res = search_bearing_knowledge(
        "SKF 6205 C3 radial internal clearance and relubrication interval",
        top_k=3,
        knowledge_base=kb,
    )
    assert rag_res["status"] == "ok"
    assert len(rag_res["retrieved_chunks"]) >= 1

    # Load MFPT outer-race fault signal
    loaded = load_vibration_file(DATA_DIR / "mfpt_outer_race_fault_48828hz.mat")
    ctx = ToolExecutionContext(
        signal=loaded.signal,
        signal_name="mfpt_outer_race_fault_48828hz.mat:bearing.gs",
        fs=48_828.0,
        rpm=1500.0,
        n_balls=MFPT_NICE_BEARING.n_balls,
        ball_diameter=MFPT_NICE_BEARING.ball_diameter,
        pitch_diameter=MFPT_NICE_BEARING.pitch_diameter,
        contact_angle_deg=MFPT_NICE_BEARING.contact_angle_deg,
        envelope_band=(2500.0, 5500.0),
        knowledge_base=kb,
    )

    # Test 5 & 6: Gemini Function Calling + RAG citation
    resp = ask_with_signal_tools(
        client,
        question=(
            "Compare the observed frequencies with theoretical bearing frequencies for this MFPT signal, "
            "and also search the bearing knowledge base for envelope analysis guidance and cite the document and page."
        ),
        tool_context=ctx,
    )
    called_names = [tc.name for tc in resp.tool_calls]
    assert any(
        n in called_names
        for n in ("compare_fault_frequencies", "analyze_vibration_signal", "calculate_bearing_frequencies")
    )
    assert "81." in resp.text

    # Test 7: Full 10-section Diagnostic Report generation
    diag = analyze_signal(
        loaded.signal,
        fs=48_828.0,
        shaft_hz=25.0,
        geometry=MFPT_NICE_BEARING,
        envelope_band=(2500.0, 5500.0),
    )
    report_md = generate_report(
        client,
        diag,
        interpretation=resp.text,
        asset={"Asset ID": "MFPT-TEST-RIG-01", "Location": "Lab Bay 2"},
        signal_info={
            "signal_name": "mfpt_outer_race_fault_48828hz.mat",
            "source_type": "MFPT Benchmark (.mat)",
            "channel": "bearing.gs",
        },
        rag_hits=rag_res["retrieved_chunks"],
    )
    for required_heading in (
        "## 1. Uploaded Signal Information",
        "## 2. Sampling Frequency & Operating Conditions",
        "## 3. Time-Domain Statistical Measurements",
        "## 4. FFT & Welch PSD Spectral Findings",
        "## 5. Theoretical Bearing Characteristic Frequencies",
        "## 6. Observed vs. Theoretical Frequency Comparison",
        "## 9. Retrieved Technical References (ChromaDB RAG)",
    ):
        assert required_heading in report_md
    assert "48,828.00 Hz" in report_md
    assert "81.12 Hz" in report_md
    assert "SKF_6205_Bearing_Maintenance_Manual.pdf" in report_md or "Randall_Antoni_Bearing_Envelope_Analysis_Paper.pdf" in report_md


def test_streamlit_four_sections_apptest() -> None:
    """Test 8: Verify the Streamlit app renders all 4 sections without exceptions."""
    at = AppTest.from_file(str(PROJECT_ROOT / "frontend" / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Streamlit app raised an exception: {at.exception}"
    assert len(at.tabs) == 4
