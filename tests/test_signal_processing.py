"""Comprehensive offline unit tests for signal processing, diagnosis, and RAG chunking."""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.io import savemat  # noqa: E402

from src.diagnosis.engine import analyze_signal  # noqa: E402
from src.rag.chunker import chunk_text  # noqa: E402
from src.signal_processing import (  # noqa: E402
    CWRU_SKF_6205_2RS,
    BearingGeometry,
    SignalLoadError,
    bearing_fault_frequencies,
    compare_fault_frequencies,
    compute_fft,
    compute_psd,
    compute_spectrogram,
    compute_stft,
    extract_dominant_frequency,
    extract_significant_peaks,
    generate_bearing_signal,
    list_mat_channels,
    load_mat_signal,
    load_signal,
    load_vibration_file,
    normalize_signal,
    plot_fft_spectrum,
    plot_psd,
    plot_spectrogram,
    plot_waveform,
    preprocess_signal,
    rpm_to_hz,
    run_signal_processing_pipeline,
    time_domain_features,
    validate_sampling_rate,
)

FS = 12_000.0
RPM = 1797.0
SHAFT = RPM / 60.0


# ------------------------------------------------------------ loaders & input validation
def test_load_signal_csv():
    data = "accel\n" + "\n".join(str(v) for v in np.random.default_rng(0).normal(size=200))
    sig, _ = load_signal(io.StringIO(data))
    assert sig.size == 200


def test_load_signal_rejects_text():
    with pytest.raises(SignalLoadError):
        load_signal(io.StringIO("a,b\nx,y\n"))


def test_load_mat_signal_and_channel_listing():
    buf = io.BytesIO()
    arr_de = np.linspace(-1.0, 1.0, 500).reshape(-1, 1)
    arr_fe = np.sin(np.linspace(0, 10, 500)).reshape(-1, 1)
    savemat(buf, {"X097_DE_time": arr_de, "X097_FE_time": arr_fe, "X097RPM": np.array([[1797.0]])})

    buf.seek(0)
    channels = list_mat_channels(buf)
    assert set(channels) == {"X097_DE_time", "X097_FE_time"}

    buf.seek(0)
    sig, meta = load_mat_signal(buf, variable="X097_FE_time")
    assert sig.shape == (500,)
    assert meta["X097RPM"] == pytest.approx(1797.0)

    buf.seek(0)
    loaded = load_vibration_file(buf, file_name="bearing_test.mat")
    assert loaded.file_format == "mat"
    assert loaded.channel == "X097_DE_time"  # prefers DE_time when variable omitted
    assert loaded.signal.size == 500


def test_never_assumes_missing_fs_or_bearing_params():
    with pytest.raises(ValueError):
        validate_sampling_rate(None)
    with pytest.raises(ValueError):
        validate_sampling_rate(-100.0)
    with pytest.raises(ValueError):
        rpm_to_hz(None)
    with pytest.raises(ValueError):
        bearing_fault_frequencies(shaft_hz=SHAFT, geometry=None)
    with pytest.raises(ValueError):
        BearingGeometry(n_balls=9, ball_diameter=40.0, pitch_diameter=39.0, contact_angle_deg=0.0).validate()
    with pytest.raises(ValueError):
        BearingGeometry(n_balls=9, ball_diameter=7.94, pitch_diameter=39.04, contact_angle_deg=90.0).validate()


# --------------------------------------------------------- preprocessing & normalization
def test_normalize_signal_methods():
    x = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    z = normalize_signal(x, method="zscore")
    assert np.mean(z) == pytest.approx(0.0, abs=1e-12)
    assert np.std(z) == pytest.approx(1.0, rel=1e-6)

    mm = normalize_signal(x, method="minmax")
    assert np.min(mm) == pytest.approx(-1.0)
    assert np.max(mm) == pytest.approx(1.0)

    pk = normalize_signal(x, method="peak")
    assert np.max(np.abs(pk)) == pytest.approx(1.0)

    rm = normalize_signal(x, method="rms")
    assert np.sqrt(np.mean(rm**2)) == pytest.approx(1.0)


def test_normalize_rejects_constant_signal():
    with pytest.raises(ValueError):
        normalize_signal(np.ones(50), method="zscore")


def test_preprocess_signal_detrend_and_bandpass():
    t = np.arange(2000) / FS
    x = 5.0 + np.sin(2 * np.pi * 2500 * t) + np.sin(2 * np.pi * 20 * t)
    proc = preprocess_signal(x, fs=FS, detrend="constant", normalize="zscore", bandpass=(1500.0, 3500.0))
    assert np.mean(proc) == pytest.approx(0.0, abs=1e-6)
    assert np.std(proc) == pytest.approx(1.0, rel=1e-3)


# ------------------------------------------------------------- time-domain indicators
def test_time_features_sine():
    t = np.arange(int(FS)) / FS
    sig = 2.0 + np.sin(2 * np.pi * 50 * t)
    feats = time_domain_features(sig)
    assert feats["mean"] == pytest.approx(2.0, abs=1e-3)
    assert feats["variance"] == pytest.approx(0.5, rel=1e-3)
    assert feats["std"] == pytest.approx(1 / np.sqrt(2), rel=1e-3)

    zero_mean_feats = time_domain_features(np.sin(2 * np.pi * 50 * t))
    assert zero_mean_feats["rms"] == pytest.approx(1 / np.sqrt(2), rel=1e-3)
    assert zero_mean_feats["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-3)
    assert zero_mean_feats["kurtosis"] == pytest.approx(1.5, rel=1e-2)


def test_time_features_rejects_nan():
    with pytest.raises(ValueError):
        time_domain_features(np.array([1.0, np.nan, 2.0]))


# ---------------------------------------------------- bearing characteristic frequencies
def test_fault_frequencies_cwru_6205_and_contact_angle():
    f = bearing_fault_frequencies(rpm=RPM, geometry=CWRU_SKF_6205_2RS)
    # Published CWRU values at 1797 RPM: BPFO ≈ 107.4 Hz, BPFI ≈ 162.2 Hz
    assert f["shaft"] == pytest.approx(SHAFT, rel=1e-6)
    assert f["BPFO"] == pytest.approx(107.4, abs=0.5)
    assert f["BPFI"] == pytest.approx(162.2, abs=0.5)

    # Verify non-zero contact angle conversion (degrees -> radians)
    geom_angled = BearingGeometry(n_balls=12, ball_diameter=10.0, pitch_diameter=50.0, contact_angle_deg=15.0)
    f_ang = bearing_fault_frequencies(rpm=1800.0, geometry=geom_angled)
    fr = 30.0
    ratio = (10.0 / 50.0) * math.cos(math.radians(15.0))
    assert f_ang["FTF"] == pytest.approx((fr / 2.0) * (1.0 - ratio))
    assert f_ang["BPFO"] == pytest.approx((12 * fr / 2.0) * (1.0 - ratio))
    assert f_ang["BPFI"] == pytest.approx((12 * fr / 2.0) * (1.0 + ratio))
    assert f_ang["BSF"] == pytest.approx((50.0 * fr / (2.0 * 10.0)) * (1.0 - ratio**2))


# ---------------------------------------- FFT, dominant freq, Welch PSD, STFT, spectrogram
def test_fft_dominant_psd_stft_and_short_signal_lengths():
    t = np.arange(int(FS)) / FS
    sig = 1.5 * np.sin(2 * np.pi * 120 * t) + 0.4 * np.sin(2 * np.pi * 360 * t)

    freqs, amp = compute_fft(sig, FS)
    dom = extract_dominant_frequency(freqs, amp)
    assert dom["frequency_hz"] == pytest.approx(120.0, abs=1.0)
    assert dom["amplitude"] == pytest.approx(1.5, abs=0.1)

    peaks = extract_significant_peaks(freqs, amp)
    assert any(abs(p["frequency_hz"] - 120.0) <= 1.0 for p in peaks)
    assert any(abs(p["frequency_hz"] - 360.0) <= 1.0 for p in peaks)

    psd_f, psd_p = compute_psd(sig, FS)
    assert psd_f[np.argmax(psd_p)] == pytest.approx(120.0, abs=25.0)

    sp_f, sp_t, zxx = compute_stft(sig, FS)
    assert zxx.shape == (sp_f.size, sp_t.size)
    sf, st, sp_pow = compute_spectrogram(sig, FS)
    assert sp_pow.shape == (sf.size, st.size)
    assert np.all(sp_pow >= 0.0)

    # Very short signal lengths should still work cleanly without crashing
    short_sig = np.sin(2 * np.pi * 50 * (np.arange(20) / 1000.0))
    pf, pp = compute_psd(short_sig, 1000.0)
    sf2, st2, sp2 = compute_spectrogram(short_sig, 1000.0)
    assert pf.size > 1 and pp.size == pf.size
    assert sp2.shape == (sf2.size, st2.size)


# -------------------------------------------------------------- fault frequency comparison
def test_compare_fault_frequencies_matches_harmonics_and_is_non_conclusive():
    theo = bearing_fault_frequencies(rpm=RPM, geometry=CWRU_SKF_6205_2RS)
    t = np.arange(int(FS)) / FS
    # Inject fundamental BPFO and 2x BPFO into the raw FFT spectrum
    sig = (
        0.05 * np.random.default_rng(7).normal(size=t.size)
        + 1.0 * np.sin(2 * np.pi * theo["BPFO"] * t)
        + 0.6 * np.sin(2 * np.pi * (2.0 * theo["BPFO"]) * t)
    )
    freqs, amp = compute_fft(sig, FS)
    report = compare_fault_frequencies(freqs, amp, theo, max_harmonics=3, tolerance_pct=2.0)

    assert "BPFO" in report.possible_matches
    bpfo_comp = report.comparisons["BPFO"]
    assert bpfo_comp.possible_match is True
    assert bpfo_comp.matched_harmonic_count >= 2
    assert abs(bpfo_comp.harmonics_checked[0].frequency_diff_hz) <= 1.5
    assert "not constitute definitive proof" in report.disclaimer.lower()
    assert "observed_peaks" in report.to_dict()


# --------------------------------------------------------- end-to-end diagnosis & pipeline
def test_outer_race_fault_detected():
    bpfo = bearing_fault_frequencies(shaft_hz=SHAFT, geometry=CWRU_SKF_6205_2RS)["BPFO"]
    sig = generate_bearing_signal(fs=FS, shaft_hz=SHAFT, fault_freq=bpfo)
    result = analyze_signal(sig, FS, SHAFT, geometry=CWRU_SKF_6205_2RS, envelope_band=(2000, 4000))
    assert result.fault_matches["BPFO"]["detected"]
    assert "Outer race" in result.suspected_faults


def test_healthy_signal_no_fault():
    sig = generate_bearing_signal(fs=FS, shaft_hz=SHAFT, fault_freq=None)
    result = analyze_signal(sig, FS, SHAFT, geometry=CWRU_SKF_6205_2RS, envelope_band=(2000, 4000))
    assert result.suspected_faults == []
    assert "BPFO" in result.to_evidence_text()


def test_pipeline_and_plotly_figures():
    bpfi = bearing_fault_frequencies(rpm=RPM, geometry=CWRU_SKF_6205_2RS)["BPFI"]
    sig = generate_bearing_signal(fs=FS, shaft_hz=SHAFT, fault_freq=bpfi)
    bundle = run_signal_processing_pipeline(
        sig, fs=FS, rpm=RPM, geometry=CWRU_SKF_6205_2RS, normalize="zscore"
    )
    summary = bundle.to_summary_dict()
    assert summary["sampling_rate_hz"] == FS
    assert "variance" in summary["time_domain_features_raw"]
    assert summary["time_domain_features_processed"]["std"] == pytest.approx(1.0, rel=1e-2)

    fig_w = plot_waveform(bundle.raw_signal, FS)
    fig_f = plot_fft_spectrum(
        *bundle.fft,
        theoretical_frequencies=bundle.theoretical_frequencies,
        observed_peaks=bundle.comparison.observed_peaks,
    )
    fig_p = plot_psd(*bundle.psd, theoretical_frequencies=bundle.theoretical_frequencies)
    fig_s = plot_spectrogram(*bundle.spectrogram)
    assert len(fig_w.data) >= 1
    assert len(fig_f.data) >= 1
    assert len(fig_p.data) >= 1
    assert len(fig_s.data) >= 1


def test_chunker_overlap():
    text = ("Bearing vibration analysis sentence. " * 200).strip()
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_streamlit_app_demo_and_files_run_smoothly():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "frontend" / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Streamlit exception on initial load: {at.exception}"
    # Verify 6 time-domain metrics + 1 KB metric rendered and no errors on page
    assert len(at.error) == 0
    assert len(at.metric) >= 6

    # Switch to Built-in demo files (.csv & .mat) and verify it runs without errors
    at.radio[0].set_value("Built-in demo files (.csv & .mat)").run()
    assert not at.exception, f"Streamlit exception on demo files: {at.exception}"
    assert len(at.error) == 0

