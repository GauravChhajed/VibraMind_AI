"""VibraMind AI - Industrial Bearing Fault Diagnosis Assistant (Streamlit Frontend).

Run from the project root:
    streamlit run frontend/app.py
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import sys
from pathlib import Path

# Make `src` importable when launched via `streamlit run frontend/app.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import DATA_DIR, KNOWLEDGE_BASE_DIR, ConfigurationError, get_settings  # noqa: E402
from src.diagnosis import (  # noqa: E402
    analyze_signal,
    ask_with_signal_tools,
    generate_report,
    interpret_diagnosis,
)
from src.llm import GeminiClient, GeminiError, ToolExecutionContext  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.rag import DOCUMENT_CATEGORIES, KnowledgeBase, search_bearing_knowledge  # noqa: E402
from src.signal_processing import (  # noqa: E402
    CWRU_SKF_6205_2RS,
    MFPT_NICE_BEARING,
    VALID_DETREND,
    VALID_NORMALIZATIONS,
    BearingGeometry,
    SignalLoadError,
    bearing_fault_frequencies,
    generate_bearing_signal,
    list_csv_channels,
    list_mat_channels,
    load_vibration_file,
    plot_envelope_spectrum,
    plot_fft_spectrum,
    plot_psd,
    plot_spectrogram,
    plot_waveform,
    rpm_to_hz,
)

logger = get_logger("frontend")
st.set_page_config(
    page_title="VibraMind AI · Industrial Bearing Diagnostics",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean industrial styling
st.markdown(
    """
    <style>
    .vm-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 1.15rem 1.5rem;
        border-radius: 0.65rem;
        border-left: 5px solid #38bdf8;
        margin-bottom: 1rem;
    }
    .vm-header h2 {
        margin: 0;
        font-size: 1.55rem;
        color: #f8fafc;
    }
    .vm-header p {
        margin: 0.25rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.92rem;
    }
    .vm-card {
        background-color: rgba(30, 41, 59, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 0.55rem;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLE_QUESTIONS = [
    "Analyse this vibration signal.",
    "Does this bearing show signs of an outer-race fault?",
    "What are the dominant frequencies?",
    "Explain the RMS and kurtosis values.",
    "Compare the observed frequencies with theoretical bearing frequencies.",
    "Is there evidence of an inner-race defect?",
    "Explain the spectrogram.",
    "What additional measurements are required?",
    "According to the bearing manual, what is the SKF 6205 C3 radial clearance and relubrication interval?",
]


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def get_gemini() -> GeminiClient | None:
    try:
        return GeminiClient()
    except (ConfigurationError, GeminiError) as exc:
        logger.warning("Gemini unavailable: %s", exc)
        return None


@st.cache_resource(show_spinner=False)
def get_knowledge_base() -> KnowledgeBase | None:
    gemini = get_gemini()
    if gemini is None:
        return None
    try:
        return KnowledgeBase(gemini)
    except Exception as exc:
        logger.error("Knowledge base unavailable: %s", exc)
        return None


def retrieve_context(query: str) -> str:
    kb = get_knowledge_base()
    if kb is None or kb.count() == 0:
        return ""
    try:
        return kb.format_context(kb.query(query))
    except GeminiError as exc:
        st.warning(f"Retrieval failed: {exc}")
        return ""


def retrieve_rag_hits(query: str, top_k: int = 4) -> list[dict]:
    kb = get_knowledge_base()
    if kb is None or kb.count() == 0:
        return []
    res = search_bearing_knowledge(query, top_k=top_k, knowledge_base=kb)
    if res.get("status") == "ok":
        return res.get("retrieved_chunks", [])
    return []


def _apply_cwru_preset() -> None:
    st.session_state["input_fs"] = 12_000.0
    st.session_state["input_rpm"] = 1797.0
    st.session_state["input_n_balls"] = int(CWRU_SKF_6205_2RS.n_balls)
    st.session_state["input_ball_d"] = float(CWRU_SKF_6205_2RS.ball_diameter)
    st.session_state["input_pitch_d"] = float(CWRU_SKF_6205_2RS.pitch_diameter)
    st.session_state["input_angle"] = float(CWRU_SKF_6205_2RS.contact_angle_deg)


def _apply_mfpt_preset(fs_val: float = 48_828.0) -> None:
    st.session_state["input_fs"] = float(fs_val)
    st.session_state["input_rpm"] = 1500.0
    st.session_state["input_n_balls"] = int(MFPT_NICE_BEARING.n_balls)
    st.session_state["input_ball_d"] = float(MFPT_NICE_BEARING.ball_diameter)
    st.session_state["input_pitch_d"] = float(MFPT_NICE_BEARING.pitch_diameter)
    st.session_state["input_angle"] = float(MFPT_NICE_BEARING.contact_angle_deg)


def _load_quick_dataset(file_name: str, fs_val: float, rpm_val: float, geom: BearingGeometry) -> None:
    """Callback to load a benchmark file from data/ and set its exact bearing parameters."""
    st.session_state["data_source_radio"] = "Built-in dataset files (.mat & .csv)"
    st.session_state["selected_demo_filename"] = file_name
    st.session_state["input_fs"] = float(fs_val)
    st.session_state["input_rpm"] = float(rpm_val)
    st.session_state["input_n_balls"] = int(geom.n_balls)
    st.session_state["input_ball_d"] = float(geom.ball_diameter)
    st.session_state["input_pitch_d"] = float(geom.pitch_diameter)
    st.session_state["input_angle"] = float(geom.contact_angle_deg)


def _record_analysis_history(
    signal_name: str,
    source_type: str,
    channel: str,
    result,
) -> None:
    history: list[dict] = st.session_state.setdefault("analysis_history", [])
    dom_hz = result.dominant_frequency["frequency_hz"] if result.dominant_frequency else 0.0
    entry = {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Signal": signal_name,
        "Source": source_type,
        "Channel": channel,
        "Samples": int(result.features.get("samples", len(result.processed_signal))),
        "fs (Hz)": round(float(result.fs), 1),
        "RPM": round(float(result.shaft_hz * 60.0), 1),
        "RMS": round(float(result.features["rms"]), 4),
        "Kurtosis": round(float(result.features["kurtosis"]), 2),
        "Crest Factor": round(float(result.features["crest_factor"]), 2),
        "Dominant FFT (Hz)": round(float(dom_hz), 2),
        "Suspected Faults": ", ".join(result.suspected_faults) if result.suspected_faults else "Healthy / None",
    }
    if not history or history[0].get("Signal") != signal_name or history[0].get("fs (Hz)") != entry["fs (Hz)"]:
        history.insert(0, entry)
        del history[25:]


# ------------------------------------------------------------------- session & sidebar
settings = get_settings()
gemini = get_gemini()
kb_instance = get_knowledge_base()
state = st.session_state
state.setdefault("chat", [])
state.setdefault("last_interaction_id", None)
state.setdefault("analysis_history", [])
state.setdefault("uploaded_signal_names", set())
state.setdefault("data_source_radio", "Built-in dataset files (.mat & .csv)")
state.setdefault("selected_demo_filename", "mfpt_outer_race_fault_48828hz.mat")

# Pre-populate sidebar inputs with the explicit SKF 6205-2RS preset on first load
state.setdefault("input_fs", 12_000.0)
state.setdefault("input_rpm", 1797.0)
state.setdefault("input_n_balls", int(CWRU_SKF_6205_2RS.n_balls))
state.setdefault("input_ball_d", float(CWRU_SKF_6205_2RS.ball_diameter))
state.setdefault("input_pitch_d", float(CWRU_SKF_6205_2RS.pitch_diameter))
state.setdefault("input_angle", float(CWRU_SKF_6205_2RS.contact_angle_deg))

with st.sidebar:
    st.title("⚙️ VibraMind AI")
    st.caption("M.Tech Industrial Bearing Fault Diagnosis Assistant")

    st.subheader("Gemini AI Engine")
    if gemini is None:
        st.error("Gemini not configured. Add `GEMINI_API_KEY` to `.env` and restart.")
    else:
        st.success(f"Online · `{settings.gemini_model}`")
        if st.button("Verify API Connection", width="stretch"):
            with st.spinner("Contacting Gemini API..."):
                ok, msg = gemini.test_connection()
            (st.success if ok else st.error)(msg)

    st.divider()
    st.subheader("Bearing & Operating Parameters")
    st.caption("All parameters are validated explicitly — never silently assumed.")

    col_pre1, col_pre2 = st.columns(2)
    col_pre1.button(
        "SKF 6205 (CWRU)",
        on_click=_apply_cwru_preset,
        width="stretch",
        help="SKF 6205-2RS: N=9, d=7.94 mm, D=39.04 mm, φ=0°, 1797 RPM, fs=12,000 Hz",
    )
    col_pre2.button(
        "MFPT NICE Bearing",
        on_click=_apply_mfpt_preset,
        width="stretch",
        help="MFPT NICE bearing: N=8, d=5.969 mm, D=31.623 mm, φ=0°, 1500 RPM, fs=48,828 Hz",
    )

    fs = st.number_input(
        "Sampling frequency fs (Hz)",
        min_value=1.0,
        max_value=2_000_000.0,
        step=1000.0,
        key="input_fs",
    )
    rpm = st.number_input(
        "Shaft rotational speed (RPM)",
        min_value=0.1,
        max_value=200_000.0,
        step=10.0,
        key="input_rpm",
    )
    n_balls = st.number_input(
        "Number of rolling elements (N)",
        min_value=1,
        max_value=200,
        step=1,
        key="input_n_balls",
    )
    ball_d = st.number_input(
        "Ball / roller diameter d (mm)",
        min_value=0.01,
        max_value=1000.0,
        step=0.1,
        key="input_ball_d",
    )
    pitch_d = st.number_input(
        "Pitch diameter D (mm)",
        min_value=0.02,
        max_value=5000.0,
        step=0.5,
        key="input_pitch_d",
    )
    angle = st.number_input(
        "Contact angle φ (degrees)",
        min_value=0.0,
        max_value=89.9,
        step=1.0,
        key="input_angle",
    )

# Validate required operating & bearing parameters explicitly
params_error: str | None = None
geometry: BearingGeometry | None = None
shaft_hz: float | None = None
theoretical_freqs: dict[str, float] | None = None

missing_fields = [
    name
    for name, val in (
        ("Sampling frequency (Hz)", fs),
        ("Shaft speed (RPM)", rpm),
        ("Number of rolling elements (N)", n_balls),
        ("Ball diameter (d)", ball_d),
        ("Pitch diameter (D)", pitch_d),
    )
    if val is None
]
if missing_fields:
    params_error = "Please enter required parameter(s) in the sidebar: " + ", ".join(missing_fields)
else:
    try:
        shaft_hz = rpm_to_hz(rpm)
        geometry = BearingGeometry(
            n_balls=int(n_balls),  # type: ignore[arg-type]
            ball_diameter=float(ball_d),  # type: ignore[arg-type]
            pitch_diameter=float(pitch_d),  # type: ignore[arg-type]
            contact_angle_deg=float(angle),
        )
        theoretical_freqs = bearing_fault_frequencies(shaft_hz=shaft_hz, geometry=geometry)
    except ValueError as exc:
        params_error = str(exc)

# Top Hero Banner
st.markdown(
    """
    <div class="vm-header">
        <h2>⚙️ VibraMind AI — Intelligent Industrial Bearing Fault Diagnosis</h2>
        <p>Multi-Domain Vibration Signal Processing · Kinematic Fault Frequency Screening · ChromaDB RAG · Google Gemini Function Calling</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Count available dataset signals and indexed RAG documents for the Dashboard
builtin_signal_files = sorted(
    [p for p in DATA_DIR.glob("*") if p.suffix.lower() in {".csv", ".mat", ".txt"}]
)
total_signals_count = len(builtin_signal_files) + len(state["uploaded_signal_names"])
docs_summary = kb_instance.list_documents() if kb_instance is not None else []
total_docs_count = len(docs_summary)
total_chunks_count = kb_instance.count() if kb_instance is not None else 0

# Main Four Sections
tab_dashboard, tab_analysis, tab_chat, tab_report = st.tabs(
    [
        "📊 1. Dashboard",
        "📈 2. Signal Analysis",
        "🤖 3. AI Diagnostic Assistant",
        "📑 4. Diagnostic Reports",
    ]
)

# ===================================================================
# SECTION 2: SIGNAL ANALYSIS (evaluated so state["diagnosis"] is ready)
# ===================================================================
with tab_analysis:
    st.subheader("📈 Vibration Signal Processing & Bearing Fault Frequency Analysis")
    if params_error:
        st.error(params_error)

    source = st.radio(
        "Select Vibration Data Source",
        [
            "Built-in dataset files (.mat & .csv)",
            "Upload MATLAB (.mat) or CSV (.csv)",
            "Synthetic bearing signal generator",
        ],
        horizontal=True,
        key="data_source_radio",
    )
    raw_signal = None
    signal_label = "none"
    signal_source_type = source
    selected_channel = "default"
    signal_tag = ""

    try:
        if source == "Synthetic bearing signal generator":
            fault = st.selectbox(
                "Simulated Bearing Condition",
                [
                    "BPFO (Outer Race Fault)",
                    "BPFI (Inner Race Fault)",
                    "BSF (Ball / Roller Fault)",
                    "FTF (Cage Fault)",
                    "Healthy",
                ],
            )
            fault_key = fault.split()[0]
            signal_label = f"synthetic_{fault_key}"
            selected_channel = "synthetic_accel"
            if fs is not None and shaft_hz is not None and theoretical_freqs is not None:
                raw_signal = generate_bearing_signal(
                    fs=float(fs),
                    duration=1.0,
                    shaft_hz=shaft_hz,
                    fault_freq=None if fault_key == "Healthy" else theoretical_freqs[fault_key],
                    resonance_hz=min(3000.0, float(fs) / 4.0),
                )
                signal_tag = f"synthetic:{fault_key}:{fs}:{rpm}:{n_balls}:{ball_d}:{pitch_d}:{angle}"
        elif source == "Built-in dataset files (.mat & .csv)":
            if not builtin_signal_files:
                st.warning("No dataset files found in `data/`.")
            else:
                file_names = [p.name for p in builtin_signal_files]
                default_idx = (
                    file_names.index(state["selected_demo_filename"])
                    if state.get("selected_demo_filename") in file_names
                    else 0
                )
                chosen_name = st.selectbox(
                    "Select benchmark vibration file from `data/` (MFPT & CWRU)",
                    file_names,
                    index=default_idx,
                )
                state["selected_demo_filename"] = chosen_name
                chosen_file = DATA_DIR / chosen_name
                is_mat = chosen_file.suffix.lower() == ".mat"
                channels = list_mat_channels(chosen_file) if is_mat else list_csv_channels(chosen_file)
                selected_channel = (
                    st.selectbox("Select signal channel / variable", channels)
                    if len(channels) > 1
                    else channels[0]
                )
                loaded = load_vibration_file(chosen_file, file_name=chosen_file.name, channel=selected_channel)
                raw_signal = loaded.signal
                signal_label = f"{chosen_file.name}:{selected_channel}"
                signal_tag = f"demofile:{chosen_file.name}:{selected_channel}:{fs}:{rpm}:{n_balls}:{ball_d}:{pitch_d}:{angle}"

                meta_info = {
                    k: v for k, v in loaded.metadata.items() if k not in {"selected_variable", "available_variables"}
                }
                st.caption(
                    f"Loaded **{len(raw_signal):,} samples** from channel `{loaded.channel}` "
                    f"in `{chosen_file.name}` ({loaded.file_format.upper()})."
                    + (f" Embedded file metadata: `{meta_info}`" if meta_info else "")
                )
        else:
            upload = st.file_uploader(
                "Upload vibration signal file (MATLAB `.mat` or `.csv` / `.txt`)",
                type=["mat", "csv", "txt"],
            )
            if upload is not None:
                state["uploaded_signal_names"].add(upload.name)
                is_mat = Path(upload.name).suffix.lower() == ".mat"
                channels = list_mat_channels(upload) if is_mat else list_csv_channels(upload)
                if not channels:
                    st.error(f"No valid numeric vibration channels found in {upload.name}.")
                else:
                    selected_channel = (
                        st.selectbox("Select signal channel / variable", channels)
                        if len(channels) > 1
                        else channels[0]
                    )
                    loaded = load_vibration_file(upload, file_name=upload.name, channel=selected_channel)
                    raw_signal = loaded.signal
                    signal_label = f"{upload.name}:{selected_channel}"
                    file_hash = hashlib.md5(upload.getvalue()[:4096]).hexdigest()[:8]
                    signal_tag = f"upload:{upload.name}:{file_hash}:{selected_channel}:{fs}:{rpm}:{n_balls}:{ball_d}:{pitch_d}:{angle}"
                    st.caption(
                        f"Loaded **{len(raw_signal):,} samples** from `{loaded.channel}` "
                        f"({loaded.file_format.upper()})."
                    )
                    meta_hints = {
                        k: v
                        for k, v in loaded.metadata.items()
                        if k not in {"selected_variable", "available_variables"}
                    }
                    if meta_hints:
                        st.caption(f"Metadata detected in file (for reference): `{meta_hints}`")
    except (SignalLoadError, ValueError) as exc:
        st.error(str(exc))

    with st.expander("⚙️ Signal Preprocessing, Normalization & Envelope Filter Settings", expanded=False):
        col_p1, col_p2, col_p3 = st.columns(3)
        detrend_mode = col_p1.selectbox("Detrend / DC Removal", VALID_DETREND, index=1)
        norm_mode = col_p2.selectbox("Signal Normalization", VALID_NORMALIZATIONS, index=0)
        use_band = col_p3.checkbox("Band-pass filter before Hilbert envelope", value=True)
        band = None
        if use_band and fs is not None and float(fs) > 10.0:
            nyq = float(fs) / 2.0
            lo, hi = st.slider(
                "Envelope demodulation passband (Hz)",
                1.0,
                float(nyq - 1.0),
                (min(2000.0, nyq * 0.25), min(4500.0, nyq * 0.65)),
            )
            band = (lo, hi) if 0 < lo < hi < nyq else None

    # Theoretical Bearing Characteristic Frequencies Banner
    if theoretical_freqs is not None and shaft_hz is not None:
        st.subheader("Theoretical Bearing Characteristic Frequencies (Zero-Slip Kinematic Model)")
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Shaft Speed (fr)", f"{shaft_hz:.2f} Hz", f"{rpm:.0f} RPM")
        b2.metric("FTF (Cage)", f"{theoretical_freqs['FTF']:.2f} Hz", f"{theoretical_freqs['FTF']/shaft_hz:.3f}× fr")
        b3.metric("BPFO (Outer Race)", f"{theoretical_freqs['BPFO']:.2f} Hz", f"{theoretical_freqs['BPFO']/shaft_hz:.3f}× fr")
        b4.metric("BPFI (Inner Race)", f"{theoretical_freqs['BPFI']:.2f} Hz", f"{theoretical_freqs['BPFI']/shaft_hz:.3f}× fr")
        b5.metric("BSF (Ball/Roller)", f"{theoretical_freqs['BSF']:.2f} Hz", f"{theoretical_freqs['BSF']/shaft_hz:.3f}× fr")

    full_signature = f"{signal_tag}|detrend={detrend_mode}|norm={norm_mode}|band={band}"
    can_run = (
        raw_signal is not None
        and params_error is None
        and fs is not None
        and shaft_hz is not None
        and geometry is not None
    )

    run_clicked = st.button("▶️ Run Complete Signal Analysis", type="primary", disabled=not can_run)
    auto_run = can_run and state.get("last_analysis_signature") != full_signature

    if can_run and (run_clicked or auto_run):
        try:
            diag_res = analyze_signal(
                raw_signal,  # type: ignore[arg-type]
                float(fs),  # type: ignore[arg-type]
                float(shaft_hz),  # type: ignore[arg-type]
                geometry,
                envelope_band=band,
                detrend=detrend_mode,
                normalize=norm_mode,
            )
            state["diagnosis"] = diag_res
            state["raw_signal"] = raw_signal
            state["signal_label"] = signal_label
            state["signal_info"] = {
                "signal_name": signal_label,
                "source_type": signal_source_type,
                "channel": selected_channel,
            }
            state["last_analysis_signature"] = full_signature
            state.pop("interpretation", None)
            _record_analysis_history(signal_label, signal_source_type, selected_channel, diag_res)
        except ValueError as exc:
            st.error(f"Analysis failed: {exc}")

    result = state.get("diagnosis")
    if result is not None and can_run:
        raw_sig = state["raw_signal"]
        feats = result.features

        st.subheader("Time-Domain Statistical Features")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Mean", f"{feats['mean']:.4f}")
        m2.metric("Variance", f"{feats['variance']:.4f}")
        m3.metric("Std Dev", f"{feats['std']:.4f}")
        m4.metric("RMS", f"{feats['rms']:.4f}")
        m5.metric("Kurtosis", f"{feats['kurtosis']:.2f}")
        m6.metric("Crest Factor", f"{feats['crest_factor']:.2f}")

        if result.dominant_frequency:
            st.info(
                f"**Dominant FFT Peak:** `{result.dominant_frequency['frequency_hz']:.2f} Hz` "
                f"(Amplitude `{result.dominant_frequency['amplitude']:.4f}`) · "
                f"**Screening Status:** "
                + (
                    f"Possible {', '.join(result.suspected_faults)} signature detected"
                    if result.suspected_faults
                    else "No strong defect harmonic pattern above screening threshold"
                )
            )

        # 1. Waveform Plot
        st.plotly_chart(
            plot_waveform(raw_sig, result.fs, title="Original Vibration Waveform (first 0.2 s)", max_seconds=0.2),
            width="stretch",
        )
        if result.normalization != "none":
            st.plotly_chart(
                plot_waveform(
                    result.processed_signal,
                    result.fs,
                    title=f"Preprocessed & Normalized Waveform ({result.normalization}, first 0.2 s)",
                    max_seconds=0.2,
                    yaxis_title="Normalized Amplitude",
                ),
                width="stretch",
            )

        # 2. FFT Spectrum with Bearing Characteristic Frequencies & Observed Peaks
        f_fft, a_fft = result.fft
        nyq_limit = float(result.fs / 2.0)
        default_vis = min(nyq_limit, max(500.0, 5.0 * result.fault_frequencies["BPFI"]))
        max_vis_freq = st.slider(
            "Maximum display frequency for FFT & Welch PSD (Hz)",
            min_value=min(50.0, nyq_limit),
            max_value=nyq_limit,
            value=default_vis,
        )
        obs_peaks = result.fft_comparison.observed_peaks if result.fft_comparison else None
        st.plotly_chart(
            plot_fft_spectrum(
                f_fft,
                a_fft,
                theoretical_frequencies=result.fault_frequencies,
                observed_peaks=obs_peaks,
                max_freq=max_vis_freq,
            ),
            width="stretch",
        )

        # 3. Welch PSD & 4. STFT Spectrogram
        col_psd, col_spec = st.columns(2)
        f_psd, p_psd = result.psd
        col_psd.plotly_chart(
            plot_psd(
                f_psd,
                p_psd,
                theoretical_frequencies=result.fault_frequencies,
                max_freq=max_vis_freq,
            ),
            width="stretch",
        )
        f_sp, t_sp, p_sp = result.spectrogram
        col_spec.plotly_chart(
            plot_spectrogram(f_sp, t_sp, p_sp, max_freq=min(nyq_limit, 8000.0)),
            width="stretch",
        )

        # 5. Demodulated Envelope Spectrum
        f_env, a_env = result.envelope
        env_limit = min(nyq_limit, 5.0 * max(result.fault_frequencies["BPFI"], result.fault_frequencies["BSF"]))
        st.plotly_chart(
            plot_envelope_spectrum(
                f_env,
                a_env,
                theoretical_frequencies=result.fault_frequencies,
                max_freq=env_limit,
            ),
            width="stretch",
        )

        # Fault Frequency Comparison Table with Highlighted Matches
        st.subheader("Fault Frequency Comparison: Observed FFT Peaks vs. Theoretical Harmonics")
        st.caption(
            "Theoretical bearing characteristic frequencies assume zero rolling-element slip. "
            "An observed spectral match is a screening indicator only and never definitive proof of a fault."
        )
        if result.fft_comparison is not None:
            comp_rows = []
            for ft_name, comp in result.fft_comparison.comparisons.items():
                fm = result.fault_matches.get(ft_name, {}).get("harmonics", [])
                env_snr = fm[0].get("snr_db", 0.0) if fm else 0.0
                for h in comp.harmonics_checked:
                    comp_rows.append(
                        {
                            "Fault Mode": ft_name,
                            "Harmonic": f"{h.harmonic_order}x",
                            "Theoretical (Hz)": round(h.theoretical_harmonic_hz, 2),
                            "Observed Peak (Hz)": round(h.matched_observed_hz, 2) if h.matched_observed_hz is not None else None,
                            "Freq Diff (Hz)": round(h.frequency_diff_hz, 2) if h.frequency_diff_hz is not None else None,
                            "Rel Diff (%)": round(h.relative_diff_pct, 2) if h.relative_diff_pct is not None else None,
                            "Envelope 1x SNR (dB)": round(env_snr, 1) if h.harmonic_order == 1 else None,
                            "Screening Match": "✅ Possible Match" if h.matched else "No",
                        }
                    )
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, width="stretch")

        st.subheader("Rule-Based Screening Findings")
        for finding in result.findings:
            st.write(f"• {finding}")

        if st.button("🧠 Generate Quick Gemini Evidence Interpretation", disabled=gemini is None):
            with st.spinner("Gemini is interpreting the signal-processing and RAG evidence..."):
                try:
                    ctx = retrieve_context("bearing fault " + " ".join(result.suspected_faults or ["diagnosis"]))
                    state["interpretation"] = interpret_diagnosis(gemini, result, ctx)
                except GeminiError as exc:
                    st.error(str(exc))
        if state.get("interpretation"):
            st.markdown("### Gemini Technical Interpretation")
            st.markdown(state["interpretation"])


# ===================================================================
# SECTION 1: DASHBOARD
# ===================================================================
with tab_dashboard:
    st.subheader("📊 System Overview & Operational Telemetry")
    st.markdown(
        """
        **VibraMind AI** integrates classical digital signal processing (time-domain statistics, FFT, Welch PSD,
        STFT spectrograms, and Hilbert envelope demodulation) with **ChromaDB Retrieval-Augmented Generation (RAG)**
        and **Google Gemini Function Calling** to provide transparent, citation-backed industrial bearing fault diagnosis.
        """
    )

    # Top KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Available / Uploaded Signals",
        total_signals_count,
        f"{len(builtin_signal_files)} built-in · {len(state['uploaded_signal_names'])} uploaded",
    )
    k2.metric(
        "Indexed Technical Documents",
        total_docs_count,
        f"{total_chunks_count} vector chunks in ChromaDB",
    )
    k3.metric(
        "Recent Signal Analyses",
        len(state["analysis_history"]),
        state["analysis_history"][0]["Suspected Faults"] if state["analysis_history"] else "Ready",
    )
    k4.metric(
        "Active AI Diagnostic Engine",
        settings.gemini_model if gemini is not None else "Offline",
        "7 Python Tools + ChromaDB RAG",
    )

    st.divider()
    st.subheader("⚡ Quick Access: Load Benchmark Bearing Datasets (MFPT & CWRU)")
    st.caption(
        "Click any benchmark preset below to automatically configure the exact sampling frequency, shaft RPM, "
        "and bearing geometry (`N`, `d`, `D`, `φ`), then view the results in **Section 2 (Signal Analysis)**."
    )
    q1, q2, q3, q4 = st.columns(4)
    q1.button(
        "🟢 MFPT Healthy Baseline (97.6 kHz)",
        on_click=_load_quick_dataset,
        args=("mfpt_baseline_healthy_97656hz.mat", 97_656.0, 1500.0, MFPT_NICE_BEARING),
        width="stretch",
    )
    q2.button(
        "🔴 MFPT Outer-Race Fault (48.8 kHz)",
        on_click=_load_quick_dataset,
        args=("mfpt_outer_race_fault_48828hz.mat", 48_828.0, 1500.0, MFPT_NICE_BEARING),
        width="stretch",
    )
    q3.button(
        "🟠 MFPT Inner-Race Fault (48.8 kHz)",
        on_click=_load_quick_dataset,
        args=("mfpt_inner_race_fault_48828hz.mat", 48_828.0, 1500.0, MFPT_NICE_BEARING),
        width="stretch",
    )
    q4.button(
        "🔵 CWRU SKF 6205 BPFO (12 kHz)",
        on_click=_load_quick_dataset,
        args=("cwru_6205_bpfo_12khz.mat", 12_000.0, 1797.0, CWRU_SKF_6205_2RS),
        width="stretch",
    )

    st.divider()
    col_hist, col_arch = st.columns([3, 2])
    with col_hist:
        st.subheader("🕒 Recent Signal Analyses")
        if state["analysis_history"]:
            st.dataframe(pd.DataFrame(state["analysis_history"]), hide_index=True, width="stretch")
        else:
            st.info("No signal analyses recorded yet in this session.")

        st.subheader("📂 Available Vibration Dataset Files (`data/`)")
        if builtin_signal_files:
            df_files = pd.DataFrame(
                [
                    {
                        "File Name": p.name,
                        "Format": p.suffix.upper().lstrip("."),
                        "Size (KB)": round(p.stat().st_size / 1024.0, 1),
                    }
                    for p in builtin_signal_files
                ]
            )
            st.dataframe(df_files, hide_index=True, width="stretch")

    with col_arch:
        st.subheader("🧭 Diagnostic Workflow")
        st.markdown(
            """
            1. **Upload Vibration Signal** (`.mat` / `.csv`) in **Signal Analysis**.
            2. **Signal Preprocessing & Normalization** (DC removal, linear detrend, z-score / RMS / peak).
            3. **Time, Frequency & Time-Frequency Analysis** (RMS, Kurtosis, Crest Factor, FFT, Welch PSD, STFT Spectrogram, Hilbert Envelope).
            4. **Bearing Characteristic Frequency Screening** (`FTF`, `BPFO`, `BPFI`, `BSF` harmonic matching).
            5. **Gemini Function Calling & ChromaDB RAG** in **AI Diagnostic Assistant**.
            6. **Downloadable Technical Report** in **Diagnostic Reports**.
            """
        )

    st.divider()
    with st.expander("📚 Knowledge Base & Technical Document Management (ChromaDB RAG)", expanded=True):
        kb = kb_instance
        if kb is None:
            st.warning("Knowledge base requires a valid `GEMINI_API_KEY`.")
        else:
            col_u1, col_u2, col_u3 = st.columns([2, 1, 1])
            doc_category = col_u1.selectbox("Document Category", DOCUMENT_CATEGORIES, index=0)
            chunk_sz = col_u2.number_input("Chunk size (chars)", min_value=300, max_value=3000, value=1000, step=100)
            chunk_ovl = col_u3.number_input("Chunk overlap (chars)", min_value=0, max_value=500, value=200, step=50)

            files = st.file_uploader(
                "Upload bearing manuals, maintenance guides, or research papers (.pdf)",
                type=["pdf"],
                accept_multiple_files=True,
            )
            if files and st.button("Extract, Chunk, Embed & Index PDF(s) in ChromaDB", type="primary"):
                KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
                for file in files:
                    dest = KNOWLEDGE_BASE_DIR / Path(file.name).name
                    dest.write_bytes(file.getvalue())
                    with st.spinner(f"Indexing {file.name} into ChromaDB..."):
                        try:
                            n = kb.add_pdf(
                                dest,
                                category=doc_category,
                                chunk_size=int(chunk_sz),
                                overlap=int(chunk_ovl),
                            )
                            st.success(f"Indexed `{file.name}` ({doc_category}): {n} chunks stored.")
                        except Exception as exc:
                            st.error(f"`{file.name}`: {exc}")
                st.rerun()

            if docs_summary:
                df_docs = pd.DataFrame(
                    [
                        {
                            "Document Name": d["document_name"],
                            "Category": d["category"],
                            "Pages": f"{d['page_count']} ({', '.join(str(p) for p in d['pages'])})",
                            "Indexed Chunks": d["chunk_count"],
                        }
                        for d in docs_summary
                    ]
                )
                st.dataframe(df_docs, hide_index=True, width="stretch")

                col_del1, col_del2, col_del3 = st.columns([2, 1, 1])
                doc_to_delete = col_del1.selectbox(
                    "Select document to remove",
                    [d["document_name"] for d in docs_summary],
                )
                if col_del2.button("Delete Selected PDF", width="stretch"):
                    removed = kb.delete_document(doc_to_delete)
                    st.info(f"Removed {removed} chunks for `{doc_to_delete}`.")
                    st.rerun()
                if col_del3.button("Re-index `knowledge_base/`", width="stretch"):
                    with st.spinner("Re-indexing all PDFs in `knowledge_base/`..."):
                        kb.index_directory(KNOWLEDGE_BASE_DIR, default_category=doc_category)
                    st.rerun()
            else:
                st.info("No PDF documents currently indexed in ChromaDB.")
                if st.button("Index Built-In Reference PDFs (`knowledge_base/`)"):
                    with st.spinner("Indexing PDFs..."):
                        kb.index_directory(KNOWLEDGE_BASE_DIR, default_category=doc_category)
                    st.rerun()


# ===================================================================
# SECTION 3: AI DIAGNOSTIC ASSISTANT (Gemini Function Calling + RAG)
# ===================================================================
with tab_chat:
    st.subheader("🤖 Conversational AI Diagnostic Assistant (Gemini Function Calling + ChromaDB RAG)")
    active_sig = raw_signal if raw_signal is not None else state.get("raw_signal")
    active_name = signal_label if raw_signal is not None else state.get("signal_label", "signal")

    tool_context = ToolExecutionContext(
        signal=active_sig,
        signal_name=active_name,
        fs=float(fs) if fs is not None else None,
        rpm=float(rpm) if rpm is not None else None,
        n_balls=int(n_balls) if n_balls is not None else None,
        ball_diameter=float(ball_d) if ball_d is not None else None,
        pitch_diameter=float(pitch_d) if pitch_d is not None else None,
        contact_angle_deg=float(angle) if angle is not None else 0.0,
        detrend=detrend_mode,
        normalize=norm_mode,
        envelope_band=band,
        knowledge_base=kb_instance,
    )

    st.caption(
        "Ask natural-language questions about your uploaded vibration signal or indexed technical manuals. "
        "Gemini automatically selects and invokes Python signal-processing tools (`analyze_vibration_signal`, "
        "`calculate_signal_features`, `perform_fft_analysis`, `calculate_psd`, `calculate_bearing_frequencies`, "
        "`compare_fault_frequencies`) and `search_bearing_knowledge(query)` to synthesize evidence-based answers with exact citations."
    )

    col_ctx, col_clr = st.columns([4, 1])
    col_ctx.info(f"**Active Signal & Tool Context:** {tool_context.describe_state()}")
    if col_clr.button("🗑️ Clear Chat", width="stretch"):
        state["chat"], state["last_interaction_id"] = [], None

    with st.expander("💡 Quick Diagnostic Prompts (click any question to ask Gemini)", expanded=len(state["chat"]) == 0):
        q_cols = st.columns(3)
        clicked_example: str | None = None
        for idx, ex_q in enumerate(EXAMPLE_QUESTIONS):
            if q_cols[idx % 3].button(ex_q, key=f"ex_q_{idx}", width="stretch", disabled=gemini is None):
                clicked_example = ex_q

    for msg in state["chat"]:
        with st.chat_message(msg["role"]):
            if msg.get("tool_calls"):
                with st.expander(f"🔧 Executed {len(msg['tool_calls'])} Python Tool(s)"):
                    for tc in msg["tool_calls"]:
                        st.markdown(f"**`{tc['name']}`** · args: `{tc['arguments']}`")
                        st.json(tc["output"])
            st.markdown(msg["content"])

    typed_question = st.chat_input(
        "Ask about the vibration signal, fault frequencies, spectrogram, or bearing manuals...",
        disabled=gemini is None,
    )
    active_question = clicked_example or typed_question

    if active_question:
        state["chat"].append({"role": "user", "content": active_question})
        st.chat_message("user").markdown(active_question)
        with st.chat_message("assistant"), st.spinner("Gemini is executing signal-processing & ChromaDB RAG tools..."):
            try:
                resp = ask_with_signal_tools(
                    gemini,  # type: ignore[arg-type]
                    question=active_question,
                    tool_context=tool_context,
                    rag_context=retrieve_context(active_question),
                    previous_interaction_id=state["last_interaction_id"],
                )
                state["last_interaction_id"] = resp.interaction_id
                serialized_calls = [
                    {"name": tc.name, "arguments": tc.arguments, "output": tc.output}
                    for tc in resp.tool_calls
                ]
                if serialized_calls:
                    with st.expander(f"🔧 Executed {len(serialized_calls)} Python Tool(s)", expanded=True):
                        for tc in serialized_calls:
                            st.markdown(f"**`{tc['name']}`** · args: `{tc['arguments']}`")
                            st.json(tc["output"])
                answer = resp.text or "_No response._"
            except GeminiError as exc:
                serialized_calls = []
                answer = f"⚠️ {exc}"
            st.markdown(answer)
        state["chat"].append(
            {"role": "assistant", "content": answer, "tool_calls": serialized_calls}
        )


# ===================================================================
# SECTION 4: DIAGNOSTIC REPORTS
# ===================================================================
with tab_report:
    st.subheader("📑 Comprehensive Industrial Bearing Diagnostic Report Generator")
    st.caption(
        "Generates a complete 10-section engineering report combining actual signal measurements, "
        "bearing characteristic frequencies, harmonic peak comparisons, retrieved ChromaDB technical references, "
        "uncertainty analysis, and maintenance recommendations."
    )
    result = state.get("diagnosis")
    if result is None:
        st.info("Run a signal analysis in **Section 2 (Signal Analysis)** first to generate a report.")
    else:
        col_r1, col_r2, col_r3 = st.columns(3)
        asset_id = col_r1.text_input("Asset / Machine ID", "PUMP-01 / Drive-End Bearing")
        location = col_r2.text_input("Plant / Location", "Rotating Machinery Test Bay A")
        operator = col_r3.text_input("Analyst / Inspector", "VibraMind AI Diagnostic System")

        if st.button("📑 Generate Complete Diagnostic Report", type="primary", disabled=gemini is None):
            with st.spinner("Retrieving technical references and synthesizing diagnostic report with Gemini..."):
                try:
                    bearing_desc = (
                        f"{geometry.n_balls} elements, d={geometry.ball_diameter} mm, "
                        f"D={geometry.pitch_diameter} mm, φ={geometry.contact_angle_deg}°"
                        if geometry is not None
                        else "Explicitly provided"
                    )
                    rag_query = (
                        "bearing fault diagnosis envelope analysis clearance lubrication maintenance "
                        + " ".join(result.suspected_faults or ["healthy baseline"])
                    )
                    rag_hits = retrieve_rag_hits(rag_query, top_k=4)
                    sig_info = state.get(
                        "signal_info",
                        {
                            "signal_name": state.get("signal_label", "Uploaded Signal"),
                            "source_type": source,
                            "channel": selected_channel,
                        },
                    )
                    report_md = generate_report(
                        gemini,
                        result,
                        interpretation=state.get("interpretation", ""),
                        asset={
                            "Asset ID": asset_id,
                            "Location": location,
                            "Analyst": operator,
                            "Bearing Geometry": bearing_desc,
                        },
                        signal_info=sig_info,
                        rag_hits=rag_hits,
                    )
                    state["report"] = report_md
                    state["report_json"] = json.dumps(
                        {
                            "generated_at": datetime.now().isoformat(),
                            "asset": {
                                "asset_id": asset_id,
                                "location": location,
                                "analyst": operator,
                                "bearing_geometry": bearing_desc,
                            },
                            "signal_info": sig_info,
                            "sampling_frequency_hz": result.fs,
                            "shaft_speed_rpm": round(result.shaft_hz * 60.0, 2),
                            "time_domain_features": result.features,
                            "dominant_fft_frequency": result.dominant_frequency,
                            "bearing_characteristic_frequencies_hz": result.fault_frequencies,
                            "suspected_faults": result.suspected_faults,
                            "screening_findings": result.findings,
                            "retrieved_references": [
                                {
                                    "document_name": h.get("document_name"),
                                    "page_number": h.get("page_number"),
                                    "category": h.get("category"),
                                    "similarity_score": h.get("similarity_score"),
                                }
                                for h in rag_hits
                            ],
                        },
                        indent=2,
                    )
                except GeminiError as exc:
                    st.error(str(exc))

        if state.get("report"):
            dl1, dl2, dl3 = st.columns(3)
            dl1.download_button(
                "⬇️ Download Markdown Report (.md)",
                state["report"],
                file_name="vibramind_diagnostic_report.md",
                mime="text/markdown",
                width="stretch",
            )
            dl2.download_button(
                "⬇️ Download Plain-Text Report (.txt)",
                state["report"],
                file_name="vibramind_diagnostic_report.txt",
                mime="text/plain",
                width="stretch",
            )
            if state.get("report_json"):
                dl3.download_button(
                    "⬇️ Download Structured Telemetry (.json)",
                    state["report_json"],
                    file_name="vibramind_diagnostic_data.json",
                    mime="application/json",
                    width="stretch",
                )
            st.divider()
            st.markdown(state["report"])
