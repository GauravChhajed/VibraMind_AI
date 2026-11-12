"""Predefined, validated Python signal-processing tools for Gemini Function Calling.

Security guarantees:
- Only the 6 whitelisted functions in ``ALLOWED_TOOLS`` can be invoked by the LLM.
- No arbitrary code execution (``eval``/``exec``/shell) is permitted.
- All tool arguments are type- and range-validated before execution.
- Missing signal data or missing operating/bearing parameters return structured
  status payloads so Gemini can ask the user for the missing information instead
  of assuming defaults or fabricating numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.logger import get_logger
from src.signal_processing.bearing_frequencies import (
    BearingGeometry,
    bearing_fault_frequencies,
    rpm_to_hz,
)
from src.signal_processing.comparison import (
    NON_CONCLUSIVE_DISCLAIMER,
    compare_fault_frequencies as _compare_fft_fault_frequencies,
)
from src.signal_processing.features import time_domain_features
from src.signal_processing.preprocessing import (
    VALID_DETREND,
    VALID_NORMALIZATIONS,
    preprocess_signal,
    validate_sampling_rate,
    validate_signal,
)
from src.signal_processing.spectrum import (
    compute_fft,
    compute_psd,
    compute_spectrogram,
    envelope_spectrum,
    extract_dominant_frequency,
    extract_significant_peaks,
    peak_near,
)

logger = get_logger("llm.tools")


@dataclass
class ToolExecutionContext:
    """Holds the currently loaded vibration signal and user-entered parameters."""

    signal: np.ndarray | None = field(default=None, repr=False)
    signal_name: str = "uploaded_signal"
    fs: float | None = None
    rpm: float | None = None
    n_balls: int | None = None
    ball_diameter: float | None = None
    pitch_diameter: float | None = None
    contact_angle_deg: float | None = 0.0
    detrend: str = "constant"
    normalize: str = "none"
    envelope_band: tuple[float, float] | None = None
    knowledge_base: Any | None = field(default=None, repr=False)

    def describe_state(self) -> str:
        """Compact summary of what is currently available in the user session."""
        if self.signal is None:
            sig_info = "No vibration signal loaded."
        else:
            sig_info = f"Loaded signal '{self.signal_name}' ({len(self.signal)} samples)."
        params = {
            "fs_hz": self.fs,
            "rpm": self.rpm,
            "n_balls": self.n_balls,
            "ball_diameter_mm": self.ball_diameter,
            "pitch_diameter_mm": self.pitch_diameter,
            "contact_angle_deg": self.contact_angle_deg,
        }
        present = ", ".join(f"{k}={v}" for k, v in params.items() if v is not None)
        missing = [k for k, v in params.items() if v is None]
        kb_info = ""
        if self.knowledge_base is not None:
            try:
                kb_info = f" ChromaDB knowledge base has {self.knowledge_base.count()} indexed chunks."
            except Exception:
                kb_info = ""
        msg = f"{sig_info} Available parameters: [{present or 'none'}]."
        if missing:
            msg += f" Missing parameters: {missing}."
        return msg + kb_info


@dataclass
class ToolCallRecord:
    """Audit record of a single tool invocation triggered by Gemini."""

    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]


# --------------------------------------------------------- parameter resolution helpers
def _resolve_signal(signal: np.ndarray | None, context: ToolExecutionContext | None) -> np.ndarray | None:
    if signal is not None:
        return signal
    if context is not None and context.signal is not None:
        return context.signal
    return None


def _resolve_param(val: Any, context: ToolExecutionContext | None, attr: str) -> Any:
    if val is not None:
        return val
    if context is not None:
        return getattr(context, attr, None)
    return None


def _check_bearing_params(
    rpm: float | None,
    n_balls: int | None,
    ball_diameter: float | None,
    pitch_diameter: float | None,
    contact_angle_deg: float | None,
    context: ToolExecutionContext | None,
) -> tuple[float | None, BearingGeometry | None, list[str]]:
    rpm_v = _resolve_param(rpm, context, "rpm")
    n_v = _resolve_param(n_balls, context, "n_balls")
    d_v = _resolve_param(ball_diameter, context, "ball_diameter")
    pitch_v = _resolve_param(pitch_diameter, context, "pitch_diameter")
    angle_v = _resolve_param(contact_angle_deg, context, "contact_angle_deg")
    if angle_v is None:
        angle_v = 0.0

    missing: list[str] = []
    if rpm_v is None:
        missing.append("rpm (shaft rotational speed in RPM)")
    if n_v is None:
        missing.append("n_balls (number of rolling elements N)")
    if d_v is None:
        missing.append("ball_diameter (ball/roller diameter d in mm)")
    if pitch_v is None:
        missing.append("pitch_diameter (pitch diameter D in mm)")

    if missing:
        return None, None, missing

    geom = BearingGeometry(
        n_balls=int(n_v),
        ball_diameter=float(d_v),
        pitch_diameter=float(pitch_v),
        contact_angle_deg=float(angle_v),
    )
    geom.validate()
    return float(rpm_v), geom, []


# --------------------------------------------------------- 6 Predefined Python Tools
def calculate_signal_features(
    detrend: str | None = None,
    normalize: str | None = None,
    *,
    signal: np.ndarray | None = None,
    fs: float | None = None,
    context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """Tool 2: Calculate time-domain features (mean, variance, std, RMS, kurtosis, crest factor, etc.)."""
    raw = _resolve_signal(signal, context)
    if raw is None:
        return {
            "status": "missing_data",
            "message": "No vibration signal is currently loaded. Ask the user to upload or select a vibration signal.",
        }

    fs_val = _resolve_param(fs, context, "fs")
    det_mode = detrend or (context.detrend if context else "none")
    norm_mode = normalize or (context.normalize if context else "none")

    try:
        x = validate_signal(raw, min_samples=2)
        raw_feats = time_domain_features(x)
        result: dict[str, Any] = {
            "status": "ok",
            "n_samples": int(x.size),
            "duration_s": float(x.size / float(fs_val)) if fs_val and float(fs_val) > 0 else None,
            "time_domain_features": {k: round(float(v), 6) for k, v in raw_feats.items()},
        }
        if det_mode != "none" or norm_mode != "none":
            eff_fs = float(fs_val) if fs_val and float(fs_val) > 0 else 1.0
            proc = preprocess_signal(x, fs=eff_fs, detrend=det_mode, normalize=norm_mode)
            result["preprocessing"] = {"detrend": det_mode, "normalize": norm_mode}
            result["time_domain_features_preprocessed"] = {
                k: round(float(v), 6) for k, v in time_domain_features(proc).items()
            }
        return result
    except Exception as exc:
        return {"status": "validation_error", "error": str(exc)}


def perform_fft_analysis(
    fs: float | None = None,
    top_k_peaks: int = 10,
    min_snr_db: float = 6.0,
    min_prominence_ratio: float = 0.05,
    *,
    signal: np.ndarray | None = None,
    context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """Tool 3: Perform FFT frequency-domain analysis and extract dominant and significant observed peaks."""
    raw = _resolve_signal(signal, context)
    if raw is None:
        return {
            "status": "missing_data",
            "message": "No vibration signal is currently loaded. Ask the user to upload or select a vibration signal.",
        }

    fs_raw = _resolve_param(fs, context, "fs")
    if fs_raw is None:
        return {
            "status": "missing_parameters",
            "missing": ["fs (sampling frequency in Hz)"],
            "message": "Sampling frequency (fs in Hz) is required for FFT analysis and cannot be assumed. Ask the user for fs.",
        }

    try:
        fs_val = validate_sampling_rate(fs_raw)
        x = validate_signal(raw, min_samples=4)
        top_k = max(1, min(int(top_k_peaks), 30))
        freqs, amp = compute_fft(x, fs_val)
        dominant = extract_dominant_frequency(freqs, amp)
        peaks = extract_significant_peaks(
            freqs,
            amp,
            max_peaks=top_k,
            min_prominence_ratio=float(min_prominence_ratio),
            min_snr_db=float(min_snr_db),
        )
        return {
            "status": "ok",
            "sampling_rate_hz": fs_val,
            "nyquist_hz": float(fs_val / 2.0),
            "frequency_resolution_hz": round(float(freqs[1] - freqs[0]), 4),
            "dominant_observed_frequency": {
                "frequency_hz": round(dominant["frequency_hz"], 3),
                "amplitude": round(dominant["amplitude"], 6),
            },
            "significant_observed_peaks": [
                {
                    "frequency_hz": round(p["frequency_hz"], 3),
                    "amplitude": round(p["amplitude"], 6),
                    "prominence": round(p["prominence"], 6),
                    "snr_db": round(p["snr_db"], 2),
                }
                for p in peaks
            ],
        }
    except Exception as exc:
        return {"status": "validation_error", "error": str(exc)}


def calculate_psd(
    fs: float | None = None,
    nperseg: int | None = None,
    top_k_peaks: int = 5,
    *,
    signal: np.ndarray | None = None,
    context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """Tool 4: Calculate Welch Power Spectral Density (PSD) and spectral band energy distribution."""
    raw = _resolve_signal(signal, context)
    if raw is None:
        return {
            "status": "missing_data",
            "message": "No vibration signal is currently loaded. Ask the user to upload or select a vibration signal.",
        }

    fs_raw = _resolve_param(fs, context, "fs")
    if fs_raw is None:
        return {
            "status": "missing_parameters",
            "missing": ["fs (sampling frequency in Hz)"],
            "message": "Sampling frequency (fs in Hz) is required for PSD calculation. Ask the user for fs.",
        }

    try:
        fs_val = validate_sampling_rate(fs_raw)
        x = validate_signal(raw, min_samples=4)
        freqs, psd_vals = compute_psd(x, fs_val, nperseg=nperseg)
        df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
        total_power = float(np.trapezoid(psd_vals, freqs)) if freqs.size > 1 else float(np.sum(psd_vals) * df)

        peak_idx = int(np.argmax(psd_vals[1:]) + 1) if psd_vals.size > 1 else 0
        nyq = fs_val / 2.0
        low_mask = freqs <= (nyq / 3.0)
        mid_mask = (freqs > (nyq / 3.0)) & (freqs <= (2.0 * nyq / 3.0))
        high_mask = freqs > (2.0 * nyq / 3.0)

        def _band_power(mask: np.ndarray) -> float:
            if not np.any(mask):
                return 0.0
            return float(np.sum(psd_vals[mask]) * df)

        top_k = max(1, min(int(top_k_peaks), 15))
        order = np.argsort(psd_vals[1:])[::-1][:top_k] + 1 if psd_vals.size > 1 else np.array([0])

        return {
            "status": "ok",
            "sampling_rate_hz": fs_val,
            "nyquist_hz": float(nyq),
            "total_integrated_power": round(total_power, 6),
            "peak_psd": {
                "frequency_hz": round(float(freqs[peak_idx]), 3),
                "psd_value": round(float(psd_vals[peak_idx]), 8),
            },
            "band_power_distribution": {
                f"low_0_to_{nyq/3:.0f}_hz": round(_band_power(low_mask), 6),
                f"mid_{nyq/3:.0f}_to_{2*nyq/3:.0f}_hz": round(_band_power(mid_mask), 6),
                f"high_{2*nyq/3:.0f}_to_{nyq:.0f}_hz": round(_band_power(high_mask), 6),
            },
            "top_psd_bins": [
                {"frequency_hz": round(float(freqs[i]), 3), "psd_value": round(float(psd_vals[i]), 8)}
                for i in order
            ],
        }
    except Exception as exc:
        return {"status": "validation_error", "error": str(exc)}


def calculate_bearing_frequencies(
    rpm: float | None = None,
    n_balls: int | None = None,
    ball_diameter: float | None = None,
    pitch_diameter: float | None = None,
    contact_angle_deg: float | None = None,
    *,
    context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """Tool 5: Calculate theoretical bearing characteristic frequencies (FTF, BPFO, BPFI, BSF)."""
    try:
        rpm_val, geom, missing = _check_bearing_params(
            rpm, n_balls, ball_diameter, pitch_diameter, contact_angle_deg, context
        )
        if missing:
            return {
                "status": "missing_parameters",
                "missing": missing,
                "message": (
                    "Cannot calculate bearing characteristic frequencies without all bearing parameters. "
                    f"Please ask the user to provide: {', '.join(missing)}."
                ),
            }
        assert rpm_val is not None and geom is not None
        freqs = bearing_fault_frequencies(rpm=rpm_val, geometry=geom)
        harmonics = {
            name: {f"{k}x_hz": round(freqs[name] * k, 3) for k in (1, 2, 3)}
            for name in ("FTF", "BPFO", "BPFI", "BSF")
        }
        return {
            "status": "ok",
            "operating_and_geometry_inputs": {
                "shaft_rpm": rpm_val,
                "shaft_frequency_hz": round(freqs["shaft"], 4),
                "n_balls": geom.n_balls,
                "ball_diameter_mm": geom.ball_diameter,
                "pitch_diameter_mm": geom.pitch_diameter,
                "contact_angle_deg": geom.contact_angle_deg,
            },
            "theoretical_frequencies_hz": {k: round(v, 3) for k, v in freqs.items()},
            "theoretical_harmonics_hz": harmonics,
            "note": "Theoretical kinematic frequencies assume zero rolling-element slip.",
        }
    except Exception as exc:
        return {"status": "validation_error", "error": str(exc)}


def compare_fault_frequencies(
    fs: float | None = None,
    rpm: float | None = None,
    n_balls: int | None = None,
    ball_diameter: float | None = None,
    pitch_diameter: float | None = None,
    contact_angle_deg: float | None = None,
    max_harmonics: int = 3,
    tolerance_pct: float = 3.0,
    *,
    signal: np.ndarray | None = None,
    context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """Tool 6: Compare observed FFT & envelope peaks against theoretical FTF, BPFO, BPFI, and BSF."""
    raw = _resolve_signal(signal, context)
    if raw is None:
        return {
            "status": "missing_data",
            "message": "No vibration signal is currently loaded. Ask the user to upload or select a vibration signal.",
        }

    fs_raw = _resolve_param(fs, context, "fs")
    missing_all: list[str] = []
    if fs_raw is None:
        missing_all.append("fs (sampling frequency in Hz)")

    try:
        rpm_val, geom, missing_bearing = _check_bearing_params(
            rpm, n_balls, ball_diameter, pitch_diameter, contact_angle_deg, context
        )
        missing_all.extend(missing_bearing)
        if missing_all:
            return {
                "status": "missing_parameters",
                "missing": missing_all,
                "message": (
                    "Cannot compare observed frequencies with theoretical bearing frequencies because "
                    f"parameters are missing: {', '.join(missing_all)}. Ask the user for them."
                ),
            }

        fs_val = validate_sampling_rate(fs_raw)
        assert rpm_val is not None and geom is not None
        x = validate_signal(raw, min_samples=16)
        n_harm = max(1, min(int(max_harmonics), 10))
        tol = float(tolerance_pct)
        if not (0.0 < tol <= 20.0):
            raise ValueError("tolerance_pct must be between 0 and 20.")

        theo = bearing_fault_frequencies(rpm=rpm_val, geometry=geom)
        fft_f, fft_a = compute_fft(x, fs_val)
        report = _compare_fft_fault_frequencies(
            fft_f,
            fft_a,
            theo,
            max_harmonics=n_harm,
            tolerance_pct=tol,
        )

        # Also check demodulated envelope spectrum around resonance band for modulation evidence
        nyq = fs_val / 2.0
        env_band = context.envelope_band if context and context.envelope_band else None
        if env_band is None and nyq > 100.0:
            env_band = (min(2000.0, nyq * 0.25), min(4000.0, nyq * 0.75))
            if not (0 < env_band[0] < env_band[1] < nyq):
                env_band = None
        env_f, env_a = envelope_spectrum(x, fs_val, band=env_band)

        envelope_checks: dict[str, Any] = {}
        for name in ("FTF", "BPFO", "BPFI", "BSF"):
            base = theo[name]
            harms = []
            for order in range(1, n_harm + 1):
                target = base * order
                if target >= nyq:
                    break
                pf, pa = peak_near(env_f, env_a, target, tolerance=tol / 100.0)
                mask = (env_f >= target - max(10.0, 0.25 * base)) & (env_f <= target + max(10.0, 0.25 * base))
                floor = float(np.median(env_a[mask])) if np.any(mask) else float(np.median(env_a))
                snr_db = float(20.0 * np.log10((pa + 1e-12) / (floor + 1e-12)))
                harms.append(
                    {
                        "harmonic_order": order,
                        "theoretical_hz": round(target, 2),
                        "observed_envelope_peak_hz": round(pf, 2),
                        "frequency_diff_hz": round(pf - target, 2),
                        "envelope_snr_db": round(snr_db, 2),
                    }
                )
            detected_env = bool(harms) and harms[0]["envelope_snr_db"] >= 10.0 and (
                len(harms) < 2 or harms[1]["envelope_snr_db"] >= 6.0
            )
            envelope_checks[name] = {
                "envelope_modulation_detected": detected_env,
                "harmonics": harms,
            }

        data = report.to_dict()
        data["status"] = "ok"
        data["envelope_band_hz"] = list(env_band) if env_band else None
        data["envelope_spectrum_checks"] = envelope_checks
        data["disclaimer"] = NON_CONCLUSIVE_DISCLAIMER
        return data
    except Exception as exc:
        return {"status": "validation_error", "error": str(exc)}


def analyze_vibration_signal(
    fs: float | None = None,
    rpm: float | None = None,
    n_balls: int | None = None,
    ball_diameter: float | None = None,
    pitch_diameter: float | None = None,
    contact_angle_deg: float | None = None,
    detrend: str = "constant",
    normalize: str = "none",
    max_harmonics: int = 3,
    tolerance_pct: float = 3.0,
    *,
    signal: np.ndarray | None = None,
    context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """Tool 1: Comprehensive vibration analysis (time-domain, FFT, PSD, spectrogram summary, and bearing fault comparison)."""
    raw = _resolve_signal(signal, context)
    if raw is None:
        return {
            "status": "missing_data",
            "message": "No vibration signal is currently loaded. Ask the user to upload or select a vibration signal.",
        }

    fs_raw = _resolve_param(fs, context, "fs")
    if fs_raw is None:
        return {
            "status": "missing_parameters",
            "missing": ["fs (sampling frequency in Hz)"],
            "message": "Sampling frequency (fs in Hz) is required to analyze the signal. Ask the user for fs.",
        }

    try:
        fs_val = validate_sampling_rate(fs_raw)
        x = validate_signal(raw, min_samples=16)

        feats = calculate_signal_features(
            detrend=detrend, normalize=normalize, signal=x, fs=fs_val, context=context
        )
        fft_res = perform_fft_analysis(fs=fs_val, top_k_peaks=10, signal=x, context=context)
        psd_res = calculate_psd(fs=fs_val, top_k_peaks=5, signal=x, context=context)

        # Compute time-frequency STFT spectrogram summary for Gemini
        sp_f, sp_t, sp_p = compute_spectrogram(x, fs_val)
        max_idx = np.unravel_index(int(np.argmax(sp_p)), sp_p.shape)
        frame_energy = np.sum(sp_p, axis=0)
        energy_cv = float(np.std(frame_energy) / (np.mean(frame_energy) + 1e-12))
        spectrogram_summary = {
            "freq_bins": int(sp_f.size),
            "time_frames": int(sp_t.size),
            "peak_time_s": round(float(sp_t[max_idx[1]]), 4),
            "peak_frequency_hz": round(float(sp_f[max_idx[0]]), 2),
            "temporal_energy_variation_cv": round(energy_cv, 3),
            "temporal_character": (
                "transient / impulsive bursts over time"
                if energy_cv > 0.35
                else "relatively stationary over the window"
            ),
        }

        rpm_val, geom, missing_bearing = _check_bearing_params(
            rpm, n_balls, ball_diameter, pitch_diameter, contact_angle_deg, context
        )
        bearing_freqs_res: dict[str, Any] | None = None
        comparison_res: dict[str, Any] | None = None
        if not missing_bearing and rpm_val is not None and geom is not None:
            bearing_freqs_res = calculate_bearing_frequencies(
                rpm=rpm_val,
                n_balls=geom.n_balls,
                ball_diameter=geom.ball_diameter,
                pitch_diameter=geom.pitch_diameter,
                contact_angle_deg=geom.contact_angle_deg,
                context=context,
            )
            comparison_res = compare_fault_frequencies(
                fs=fs_val,
                rpm=rpm_val,
                n_balls=geom.n_balls,
                ball_diameter=geom.ball_diameter,
                pitch_diameter=geom.pitch_diameter,
                contact_angle_deg=geom.contact_angle_deg,
                max_harmonics=max_harmonics,
                tolerance_pct=tolerance_pct,
                signal=x,
                context=context,
            )

        return {
            "status": "ok",
            "signal_name": context.signal_name if context else "signal",
            "sampling_rate_hz": fs_val,
            "n_samples": int(x.size),
            "duration_s": round(float(x.size / fs_val), 4),
            "time_domain": feats.get("time_domain_features"),
            "fft_analysis": {
                "dominant_observed_frequency": fft_res.get("dominant_observed_frequency"),
                "significant_observed_peaks": fft_res.get("significant_observed_peaks"),
            },
            "psd_analysis": {
                "total_integrated_power": psd_res.get("total_integrated_power"),
                "peak_psd": psd_res.get("peak_psd"),
                "band_power_distribution": psd_res.get("band_power_distribution"),
            },
            "spectrogram_summary": spectrogram_summary,
            "bearing_frequencies": bearing_freqs_res,
            "fault_frequency_comparison": comparison_res,
            "missing_bearing_parameters": missing_bearing,
            "disclaimer": NON_CONCLUSIVE_DISCLAIMER,
        }
    except Exception as exc:
        return {"status": "validation_error", "error": str(exc)}


def search_bearing_knowledge(
    query: str,
    top_k: int = 4,
    document_name: str | None = None,
    *,
    knowledge_base: Any | None = None,
    context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """Tool 7 (RAG): Semantic search over ChromaDB bearing manuals, maintenance docs, and papers."""
    from src.rag.vector_store import search_bearing_knowledge as _rag_search

    kb = knowledge_base or (context.knowledge_base if context is not None else None)
    return _rag_search(query=query, top_k=top_k, document_name=document_name, knowledge_base=kb)


# ------------------------------------------------ Gemini Function Declarations
TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "analyze_vibration_signal",
        "description": (
            "Run a comprehensive vibration analysis on the currently loaded signal: "
            "time-domain indicators, FFT dominant & significant observed peaks, Welch PSD summary, "
            "STFT spectrogram summary, theoretical bearing frequencies, and fault frequency comparison."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fs": {"type": "number", "description": "Sampling frequency in Hz (if provided by user)."},
                "rpm": {"type": "number", "description": "Shaft rotational speed in RPM."},
                "n_balls": {"type": "integer", "description": "Number of rolling elements (N)."},
                "ball_diameter": {"type": "number", "description": "Ball or roller diameter (d) in mm."},
                "pitch_diameter": {"type": "number", "description": "Bearing pitch diameter (D) in mm."},
                "contact_angle_deg": {"type": "number", "description": "Contact angle (phi) in degrees."},
                "detrend": {
                    "type": "string",
                    "enum": list(VALID_DETREND),
                    "description": "Detrending mode ('constant', 'linear', or 'none').",
                },
                "normalize": {
                    "type": "string",
                    "enum": list(VALID_NORMALIZATIONS),
                    "description": "Normalization mode ('none', 'zscore', 'minmax', 'peak', 'rms').",
                },
                "max_harmonics": {"type": "integer", "description": "Number of fault harmonics to check (1..10)."},
                "tolerance_pct": {"type": "number", "description": "Matching tolerance percentage (e.g. 3.0)."},
            },
        },
    },
    {
        "type": "function",
        "name": "calculate_signal_features",
        "description": (
            "Calculate time-domain statistical condition indicators of the loaded vibration signal: "
            "mean, variance, standard deviation (std), RMS, peak, peak-to-peak, crest factor, "
            "kurtosis (Pearson, Gaussian=3), skewness, shape factor, and impulse factor."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "detrend": {
                    "type": "string",
                    "enum": list(VALID_DETREND),
                    "description": "Optional detrending mode before feature calculation.",
                },
                "normalize": {
                    "type": "string",
                    "enum": list(VALID_NORMALIZATIONS),
                    "description": "Optional normalization method before feature calculation.",
                },
            },
        },
    },
    {
        "type": "function",
        "name": "perform_fft_analysis",
        "description": (
            "Compute the single-sided FFT amplitude spectrum of the loaded vibration signal, "
            "extract the dominant non-DC frequency, and list significant observed spectral peaks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fs": {"type": "number", "description": "Sampling frequency in Hz."},
                "top_k_peaks": {"type": "integer", "description": "Maximum number of significant peaks to return (1..30)."},
                "min_snr_db": {"type": "number", "description": "Minimum peak SNR in dB above median noise floor."},
            },
        },
    },
    {
        "type": "function",
        "name": "calculate_psd",
        "description": (
            "Compute the Welch Power Spectral Density (PSD) of the loaded vibration signal, "
            "returning total integrated power, peak PSD frequency, and low/mid/high band power distribution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fs": {"type": "number", "description": "Sampling frequency in Hz."},
                "top_k_peaks": {"type": "integer", "description": "Number of top PSD bins to return (1..15)."},
            },
        },
    },
    {
        "type": "function",
        "name": "calculate_bearing_frequencies",
        "description": (
            "Calculate theoretical kinematic bearing defect frequencies (FTF, BPFO, BPFI, BSF) "
            "and their harmonics from shaft speed (RPM) and bearing geometry (N, d, D, contact angle phi)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "rpm": {"type": "number", "description": "Shaft rotational speed in RPM."},
                "n_balls": {"type": "integer", "description": "Number of rolling elements (N)."},
                "ball_diameter": {"type": "number", "description": "Ball or roller diameter (d) in mm."},
                "pitch_diameter": {"type": "number", "description": "Bearing pitch diameter (D) in mm."},
                "contact_angle_deg": {"type": "number", "description": "Contact angle (phi) in degrees."},
            },
        },
    },
    {
        "type": "function",
        "name": "compare_fault_frequencies",
        "description": (
            "Extract significant observed peaks from the FFT spectrum and envelope spectrum of the "
            "loaded vibration signal and compare them against theoretical FTF, BPFO, BPFI, and BSF "
            "and their harmonics, returning frequency differences and screening match status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fs": {"type": "number", "description": "Sampling frequency in Hz."},
                "rpm": {"type": "number", "description": "Shaft rotational speed in RPM."},
                "n_balls": {"type": "integer", "description": "Number of rolling elements (N)."},
                "ball_diameter": {"type": "number", "description": "Ball or roller diameter (d) in mm."},
                "pitch_diameter": {"type": "number", "description": "Bearing pitch diameter (D) in mm."},
                "contact_angle_deg": {"type": "number", "description": "Contact angle (phi) in degrees."},
                "max_harmonics": {"type": "integer", "description": "Number of harmonics to check (default 3)."},
                "tolerance_pct": {"type": "number", "description": "Relative matching tolerance in percent (default 3.0)."},
            },
        },
    },
    {
        "type": "function",
        "name": "search_bearing_knowledge",
        "description": (
            "Search the ChromaDB knowledge base (industrial bearing manuals, maintenance guides, "
            "research papers, and vibration analysis reference documents) using semantic similarity "
            "to retrieve relevant chunks with actual document names, page numbers, and chunk IDs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Technical search query (e.g. 'SKF 6205 radial clearance and lubrication interval' or 'outer race defect envelope demodulation').",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant document chunks to retrieve (1..10, default 4).",
                },
                "document_name": {
                    "type": "string",
                    "description": "Optional exact document filename filter.",
                },
            },
            "required": ["query"],
        },
    },
]

ALLOWED_TOOLS = {
    "analyze_vibration_signal": (
        analyze_vibration_signal,
        {"fs", "rpm", "n_balls", "ball_diameter", "pitch_diameter", "contact_angle_deg", "detrend", "normalize", "max_harmonics", "tolerance_pct"},
    ),
    "calculate_signal_features": (
        calculate_signal_features,
        {"detrend", "normalize"},
    ),
    "perform_fft_analysis": (
        perform_fft_analysis,
        {"fs", "top_k_peaks", "min_snr_db", "min_prominence_ratio"},
    ),
    "calculate_psd": (
        calculate_psd,
        {"fs", "nperseg", "top_k_peaks"},
    ),
    "calculate_bearing_frequencies": (
        calculate_bearing_frequencies,
        {"rpm", "n_balls", "ball_diameter", "pitch_diameter", "contact_angle_deg"},
    ),
    "compare_fault_frequencies": (
        compare_fault_frequencies,
        {"fs", "rpm", "n_balls", "ball_diameter", "pitch_diameter", "contact_angle_deg", "max_harmonics", "tolerance_pct"},
    ),
    "search_bearing_knowledge": (
        search_bearing_knowledge,
        {"query", "top_k", "document_name"},
    ),
}


def execute_tool_call(
    name: str,
    arguments: dict[str, Any] | None,
    context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """Validate and execute a single whitelisted tool call safely."""
    if name not in ALLOWED_TOOLS:
        logger.warning("Rejected call to non-whitelisted tool: %s", name)
        return {
            "status": "error",
            "error": f"Tool '{name}' is not permitted. Allowed tools: {sorted(ALLOWED_TOOLS)}",
        }

    func, allowed_keys = ALLOWED_TOOLS[name]
    raw_args = dict(arguments or {})
    unexpected = set(raw_args) - allowed_keys
    if unexpected:
        logger.warning("Rejected unexpected arguments for %s: %s", name, unexpected)
        return {
            "status": "validation_error",
            "error": f"Unexpected argument(s) for '{name}': {sorted(unexpected)}",
        }

    logger.info("Executing tool '%s' with args=%s", name, raw_args)
    try:
        return func(**raw_args, context=context)
    except Exception as exc:
        logger.error("Tool '%s' raised an exception: %s", name, exc)
        return {"status": "error", "error": str(exc)}

