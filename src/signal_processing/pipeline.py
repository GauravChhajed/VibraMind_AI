"""High-level modular signal processing pipeline for UI and Gemini tool integration."""

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
from src.signal_processing.comparison import FaultComparisonReport, compare_fault_frequencies
from src.signal_processing.features import time_domain_features
from src.signal_processing.preprocessing import (
    preprocess_signal,
    validate_sampling_rate,
    validate_signal,
)
from src.signal_processing.spectrum import (
    compute_fft,
    compute_psd,
    compute_spectrogram,
    extract_dominant_frequency,
)

logger = get_logger("signal.pipeline")


@dataclass
class SignalAnalysisBundle:
    """Complete output of the modular signal-processing pipeline."""

    fs: float
    shaft_rpm: float
    shaft_hz: float
    n_samples: int
    duration_s: float
    detrend_mode: str
    normalization_method: str
    raw_signal: np.ndarray = field(repr=False)
    processed_signal: np.ndarray = field(repr=False)
    time_features_raw: dict[str, float]
    time_features_processed: dict[str, float]
    dominant_frequency: dict[str, float]
    theoretical_frequencies: dict[str, float]
    fft: tuple[np.ndarray, np.ndarray] = field(repr=False)
    psd: tuple[np.ndarray, np.ndarray] = field(repr=False)
    spectrogram: tuple[np.ndarray, np.ndarray, np.ndarray] = field(repr=False)
    comparison: FaultComparisonReport

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary suitable for Gemini tool responses."""
        return {
            "sampling_rate_hz": self.fs,
            "shaft_rpm": self.shaft_rpm,
            "shaft_hz": self.shaft_hz,
            "n_samples": self.n_samples,
            "duration_s": self.duration_s,
            "preprocessing": {
                "detrend": self.detrend_mode,
                "normalization": self.normalization_method,
            },
            "time_domain_features_raw": dict(self.time_features_raw),
            "time_domain_features_processed": dict(self.time_features_processed),
            "dominant_frequency_observed": dict(self.dominant_frequency),
            "theoretical_bearing_frequencies_hz": dict(self.theoretical_frequencies),
            "fault_frequency_comparison": self.comparison.to_dict(),
        }


def run_signal_processing_pipeline(
    signal: np.ndarray,
    fs: float,
    rpm: float,
    geometry: BearingGeometry,
    *,
    detrend: str = "constant",
    normalize: str = "none",
    bandpass: tuple[float, float] | None = None,
    max_harmonics: int = 3,
    tolerance_pct: float = 3.0,
) -> SignalAnalysisBundle:
    """Run validation, preprocessing, time/frequency/time-frequency analysis, and comparison.

    Never assumes missing ``fs``, ``rpm``, or ``geometry``.
    """
    fs_val = validate_sampling_rate(fs)
    raw = validate_signal(signal, min_samples=16)
    shaft_hz = rpm_to_hz(rpm)
    theoretical = bearing_fault_frequencies(shaft_hz=shaft_hz, geometry=geometry)

    processed = preprocess_signal(
        raw, fs=fs_val, detrend=detrend, normalize=normalize, bandpass=bandpass
    )
    feats_raw = time_domain_features(raw)
    feats_proc = time_domain_features(processed)

    fft_freqs, fft_amp = compute_fft(processed, fs_val)
    dominant = extract_dominant_frequency(fft_freqs, fft_amp)
    psd_freqs, psd_vals = compute_psd(processed, fs_val)
    spec_f, spec_t, spec_p = compute_spectrogram(processed, fs_val)

    comparison = compare_fault_frequencies(
        fft_freqs,
        fft_amp,
        theoretical,
        max_harmonics=max_harmonics,
        tolerance_pct=tolerance_pct,
    )

    logger.info(
        "Pipeline complete: %d samples @ %.1f Hz, dom=%.2f Hz, possible_matches=%s",
        raw.size,
        fs_val,
        dominant["frequency_hz"],
        comparison.possible_matches,
    )
    return SignalAnalysisBundle(
        fs=fs_val,
        shaft_rpm=float(rpm),
        shaft_hz=shaft_hz,
        n_samples=int(raw.size),
        duration_s=float(raw.size / fs_val),
        detrend_mode=detrend,
        normalization_method=normalize,
        raw_signal=raw,
        processed_signal=processed,
        time_features_raw=feats_raw,
        time_features_processed=feats_proc,
        dominant_frequency=dominant,
        theoretical_frequencies=theoretical,
        fft=(fft_freqs, fft_amp),
        psd=(psd_freqs, psd_vals),
        spectrogram=(spec_f, spec_t, spec_p),
        comparison=comparison,
    )

