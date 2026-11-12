"""Unit and live integration tests for Gemini Function Calling and predefined Python tools."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.diagnosis.interpreter import ask_with_signal_tools  # noqa: E402
from src.llm import (  # noqa: E402
    ALLOWED_TOOLS,
    GeminiClient,
    ToolExecutionContext,
    analyze_vibration_signal,
    calculate_bearing_frequencies,
    calculate_psd,
    calculate_signal_features,
    compare_fault_frequencies,
    execute_tool_call,
    perform_fft_analysis,
)
from src.signal_processing import (  # noqa: E402
    CWRU_SKF_6205_2RS,
    bearing_fault_frequencies,
    generate_bearing_signal,
    load_vibration_file,
)

FS = 12_000.0
RPM = 1797.0
SHAFT = RPM / 60.0

requires_key = pytest.mark.skipif(not get_settings().has_api_key, reason="GEMINI_API_KEY not configured")


def _make_bpfo_context() -> ToolExecutionContext:
    bpfo = bearing_fault_frequencies(rpm=RPM, geometry=CWRU_SKF_6205_2RS)["BPFO"]
    sig = generate_bearing_signal(fs=FS, shaft_hz=SHAFT, fault_freq=bpfo)
    return ToolExecutionContext(
        signal=sig,
        signal_name="test_bpfo_signal",
        fs=FS,
        rpm=RPM,
        n_balls=CWRU_SKF_6205_2RS.n_balls,
        ball_diameter=CWRU_SKF_6205_2RS.ball_diameter,
        pitch_diameter=CWRU_SKF_6205_2RS.pitch_diameter,
        contact_angle_deg=CWRU_SKF_6205_2RS.contact_angle_deg,
    )


# ------------------------------------------------ Offline tests for all 7 tools & security
def test_all_tools_registered():
    expected = {
        "analyze_vibration_signal",
        "calculate_signal_features",
        "perform_fft_analysis",
        "calculate_psd",
        "calculate_bearing_frequencies",
        "compare_fault_frequencies",
        "search_bearing_knowledge",
    }
    assert set(ALLOWED_TOOLS.keys()) == expected


def test_tool_1_analyze_vibration_signal():
    ctx = _make_bpfo_context()
    res = analyze_vibration_signal(context=ctx)
    assert res["status"] == "ok"
    assert res["sampling_rate_hz"] == FS
    assert "rms" in res["time_domain"]
    assert "kurtosis" in res["time_domain"]
    assert "dominant_observed_frequency" in res["fft_analysis"]
    assert "total_integrated_power" in res["psd_analysis"]
    assert "peak_frequency_hz" in res["spectrogram_summary"]
    assert "BPFO" in res["fault_frequency_comparison"]["possible_matches"]


def test_tool_2_calculate_signal_features():
    ctx = _make_bpfo_context()
    res = calculate_signal_features(detrend="constant", normalize="zscore", context=ctx)
    assert res["status"] == "ok"
    feats = res["time_domain_features"]
    for key in ("mean", "variance", "std", "rms", "kurtosis", "crest_factor"):
        assert key in feats
    assert res["time_domain_features_preprocessed"]["std"] == pytest.approx(1.0, rel=1e-2)


def test_tool_3_perform_fft_analysis():
    ctx = _make_bpfo_context()
    res = perform_fft_analysis(top_k_peaks=8, context=ctx)
    assert res["status"] == "ok"
    assert res["nyquist_hz"] == FS / 2.0
    assert len(res["significant_observed_peaks"]) >= 1


def test_tool_4_calculate_psd():
    ctx = _make_bpfo_context()
    res = calculate_psd(top_k_peaks=5, context=ctx)
    assert res["status"] == "ok"
    assert res["total_integrated_power"] > 0
    assert len(res["top_psd_bins"]) == 5


def test_tool_5_calculate_bearing_frequencies():
    ctx = _make_bpfo_context()
    res = calculate_bearing_frequencies(context=ctx)
    assert res["status"] == "ok"
    theo = res["theoretical_frequencies_hz"]
    assert theo["BPFO"] == pytest.approx(107.4, abs=0.5)
    assert theo["BPFI"] == pytest.approx(162.2, abs=0.5)


def test_tool_6_compare_fault_frequencies():
    ctx = _make_bpfo_context()
    res = compare_fault_frequencies(max_harmonics=3, tolerance_pct=3.0, context=ctx)
    assert res["status"] == "ok"
    assert "BPFO" in res["possible_matches"]
    assert res["envelope_spectrum_checks"]["BPFO"]["envelope_modulation_detected"] is True
    assert "not constitute definitive proof" in res["disclaimer"].lower()


def test_tools_report_missing_data_and_missing_parameters():
    empty_ctx = ToolExecutionContext(signal=None)
    assert calculate_signal_features(context=empty_ctx)["status"] == "missing_data"

    no_params_ctx = ToolExecutionContext(signal=np.ones(200), fs=None, rpm=None)
    fft_res = perform_fft_analysis(context=no_params_ctx)
    assert fft_res["status"] == "missing_parameters"
    assert any("fs" in m for m in fft_res["missing"])

    bf_res = calculate_bearing_frequencies(context=no_params_ctx)
    assert bf_res["status"] == "missing_parameters"
    assert len(bf_res["missing"]) >= 4


def test_security_rejects_unlisted_tools_and_unexpected_args():
    ctx = _make_bpfo_context()
    bad_tool = execute_tool_call("os.system", {"cmd": "dir"}, context=ctx)
    assert bad_tool["status"] == "error"
    assert "not permitted" in bad_tool["error"]

    bad_args = execute_tool_call(
        "calculate_signal_features", {"__import__": "os", "eval": "1+1"}, context=ctx
    )
    assert bad_args["status"] == "validation_error"
    assert "Unexpected argument" in bad_args["error"]


# ----------------------------------------- Live Gemini Function Calling workflow tests
@requires_key
def test_live_gemini_function_calling_workflow_on_real_file():
    data_file = Path(__file__).resolve().parent.parent / "data" / "demo_outer_race_bpfo.csv"
    loaded = load_vibration_file(data_file, file_name=data_file.name)
    ctx = ToolExecutionContext(
        signal=loaded.signal,
        signal_name=data_file.name,
        fs=FS,
        rpm=RPM,
        n_balls=CWRU_SKF_6205_2RS.n_balls,
        ball_diameter=CWRU_SKF_6205_2RS.ball_diameter,
        pitch_diameter=CWRU_SKF_6205_2RS.pitch_diameter,
        contact_angle_deg=CWRU_SKF_6205_2RS.contact_angle_deg,
    )
    gemini = GeminiClient()

    # Turn 1: Ask about outer-race fault, RMS & kurtosis, and frequency comparison
    resp1 = ask_with_signal_tools(
        gemini,
        question=(
            "Does this bearing show signs of an outer-race fault? "
            "Explain the RMS and kurtosis values and compare the observed frequencies "
            "with theoretical bearing frequencies."
        ),
        tool_context=ctx,
    )
    assert resp1.text, "Expected non-empty interpretation from Gemini"
    assert len(resp1.tool_calls) >= 1, "Expected Gemini to invoke at least one predefined Python tool"
    called_names = {tc.name for tc in resp1.tool_calls}
    assert called_names.issubset(set(ALLOWED_TOOLS.keys()))

    # Follow-up Turn 2 (multi-turn context via previous_interaction_id)
    resp2 = ask_with_signal_tools(
        gemini,
        question="What is the Welch PSD total power and what additional measurements are required?",
        tool_context=ctx,
        previous_interaction_id=resp1.interaction_id,
    )
    assert resp2.text, "Expected non-empty follow-up response from Gemini"
    assert resp2.interaction_id is not None


@requires_key
def test_live_gemini_asks_for_missing_parameters():
    ctx_missing = ToolExecutionContext(
        signal=np.sin(np.linspace(0, 50, 500)),
        signal_name="unknown_bearing_signal",
        fs=12_000.0,
        rpm=None,
        n_balls=None,
        ball_diameter=None,
        pitch_diameter=None,
    )
    gemini = GeminiClient()
    resp = ask_with_signal_tools(
        gemini,
        question="Compare the observed frequencies with theoretical bearing frequencies for this bearing.",
        tool_context=ctx_missing,
    )
    assert resp.text
    # Either Gemini called compare_fault_frequencies / calculate_bearing_frequencies (which returned missing_parameters)
    # or directly asked for the missing bearing parameters shown in the session context.
    lower = resp.text.lower()
    assert any(tok in lower for tok in ("rpm", "pitch", "ball", "rolling", "parameter"))

