"""Deterministic, rule-based bearing analysis (no ML).

Produces structured evidence that the LLM later interprets. Thresholds are
deliberately simple screening heuristics, not certified limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.logger import get_logger
from src.signal_processing.bearing_frequencies import BearingGeometry, bearing_fault_frequencies
from src.signal_processing.comparison import FaultComparisonReport, compare_fault_frequencies
from src.signal_processing.features import time_domain_features
from src.signal_processing.preprocessing import preprocess_signal, validate_sampling_rate, validate_signal
from src.signal_processing.spectrum import (
    compute_fft,
    compute_psd,
    compute_spectrogram,
    envelope_spectrum,
    extract_dominant_frequency,
    peak_near,
)

logger = get_logger("diagnosis.engine")

KURTOSIS_ALERT = 4.0  # Gaussian = 3
CREST_FACTOR_ALERT = 5.0
SNR_MATCH_DB = 10.0  # fundamental must exceed local (median) noise floor by this much
SNR_HARMONIC_DB = 6.0  # 2nd harmonic must also rise above the floor (rejects random noise peaks)
FAULT_LABELS = {"BPFO": "Outer race", "BPFI": "Inner race", "BSF": "Rolling element (ball)", "FTF": "Cage"}


@dataclass
class DiagnosisResult:
    fs: float
    shaft_hz: float
    features: dict[str, float]
    fault_frequencies: dict[str, float]
    fault_matches: dict[str, dict]
    findings: list[str]
    suspected_faults: list[str]
    fft: tuple[np.ndarray, np.ndarray] = field(repr=False)
    envelope: tuple[np.ndarray, np.ndarray] = field(repr=False)
    psd: tuple[np.ndarray, np.ndarray] = field(repr=False, default_factory=lambda: (np.array([]), np.array([])))
    spectrogram: tuple[np.ndarray, np.ndarray, np.ndarray] = field(
        repr=False, default_factory=lambda: (np.array([]), np.array([]), np.empty((0, 0)))
    )
    dominant_frequency: dict[str, float] = field(default_factory=dict)
    fft_comparison: FaultComparisonReport | None = None
    processed_signal: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    normalization: str = "none"
    detrend: str = "constant"
    geometry: BearingGeometry | None = None

    def to_evidence_text(self) -> str:
        """Compact text representation used in LLM prompts."""
        f = self.features
        dom = self.dominant_frequency
        dom_str = f"{dom['frequency_hz']:.2f} Hz (amp {dom['amplitude']:.4f})" if dom else "N/A"
        lines = [
            f"- Sampling rate: {self.fs:.0f} Hz; shaft speed: {self.shaft_hz:.2f} Hz ({self.shaft_hz * 60:.0f} RPM)",
            f"- Normalization applied: {self.normalization}",
            f"- Mean: {f['mean']:.4f}, Variance: {f['variance']:.4f}, Std: {f['std']:.4f}, "
            f"RMS: {f['rms']:.4f}, Crest factor: {f['crest_factor']:.2f}, Kurtosis: {f['kurtosis']:.2f}",
            f"- Observed dominant FFT frequency: {dom_str}",
            "- Theoretical bearing frequencies (Hz, kinematic reference): "
            + ", ".join(f"{k}={v:.2f}" for k, v in self.fault_frequencies.items()),
            "- Envelope-spectrum matches (fundamental and harmonics, SNR dB above local floor):",
        ]
        for name, m in self.fault_matches.items():
            harmonics = ", ".join(f"{h['order']}x@{h['freq']:.1f}Hz={h['snr_db']:.1f}dB" for h in m["harmonics"])
            lines.append(f"  - {name}: {'DETECTED' if m['detected'] else 'not detected'} ({harmonics})")
        if self.fft_comparison is not None:
            possible = ", ".join(self.fft_comparison.possible_matches) or "none"
            lines.append(
                f"- FFT observed-peak matches against theoretical frequencies (screening only, not proof): {possible}"
            )
        lines.append("- Rule-based findings: " + ("; ".join(self.findings) or "none"))
        return "\n".join(lines)


def _local_noise_floor(freqs: np.ndarray, amp: np.ndarray, target: float, half_width: float) -> float:
    mask = (freqs >= target - half_width) & (freqs <= target + half_width)
    return float(np.median(amp[mask])) if mask.any() else float(np.median(amp))


def analyze_signal(
    signal: np.ndarray,
    fs: float,
    shaft_hz: float,
    geometry: BearingGeometry | None = None,
    envelope_band: tuple[float, float] | None = None,
    n_harmonics: int = 3,
    detrend: str = "constant",
    normalize: str = "none",
) -> DiagnosisResult:
    """Run the full signal-processing pipeline and rule-based screening.

    Requires explicit ``fs``, ``shaft_hz``, and ``geometry`` (never assumed).
    """
    fs_val = validate_sampling_rate(fs)
    raw = validate_signal(signal, min_samples=16)
    if geometry is None:
        raise ValueError("BearingGeometry must be explicitly provided and cannot be assumed.")

    logger.info("Analysing signal: %d samples @ %.0f Hz, shaft %.2f Hz", len(raw), fs_val, shaft_hz)
    processed = preprocess_signal(raw, fs=fs_val, detrend=detrend, normalize=normalize)
    features = time_domain_features(processed)
    fault_freqs = bearing_fault_frequencies(shaft_hz=shaft_hz, geometry=geometry)

    fft_f, fft_a = compute_fft(processed, fs_val)
    dominant = extract_dominant_frequency(fft_f, fft_a)
    psd_f, psd_p = compute_psd(processed, fs_val)
    spec_f, spec_t, spec_p = compute_spectrogram(processed, fs_val)
    fft_comparison = compare_fault_frequencies(fft_f, fft_a, fault_freqs, max_harmonics=n_harmonics)
    env_f, env_a = envelope_spectrum(processed, fs_val, band=envelope_band)

    matches: dict[str, dict] = {}
    for name in ("BPFO", "BPFI", "BSF", "FTF"):
        base = fault_freqs[name]
        harmonics = []
        for order in range(1, n_harmonics + 1):
            target = base * order
            if target >= fs_val / 2:
                break
            pf, pa = peak_near(env_f, env_a, target)
            floor = _local_noise_floor(env_f, env_a, target, half_width=max(10.0, 0.25 * base))
            snr_db = 20 * np.log10((pa + 1e-12) / (floor + 1e-12))
            harmonics.append({"order": order, "freq": pf, "amp": pa, "snr_db": float(snr_db)})
        detected = bool(harmonics) and harmonics[0]["snr_db"] >= SNR_MATCH_DB and (
            len(harmonics) < 2 or harmonics[1]["snr_db"] >= SNR_HARMONIC_DB
        )
        matches[name] = {"detected": detected, "harmonics": harmonics}

    findings: list[str] = []
    if features["kurtosis"] > KURTOSIS_ALERT:
        findings.append(f"High kurtosis ({features['kurtosis']:.2f}) indicates impulsive content")
    if features["crest_factor"] > CREST_FACTOR_ALERT:
        findings.append(f"High crest factor ({features['crest_factor']:.2f})")
    suspected = [FAULT_LABELS[n] for n, m in matches.items() if m["detected"]]
    for name, m in matches.items():
        if m["detected"]:
            findings.append(f"{name} ({fault_freqs[name]:.1f} Hz) aligned in envelope spectrum (screening indicator)")
    if not findings:
        findings.append("No significant fault indicators detected")

    return DiagnosisResult(
        fs=fs_val,
        shaft_hz=shaft_hz,
        features=features,
        fault_frequencies=fault_freqs,
        fault_matches=matches,
        findings=findings,
        suspected_faults=suspected,
        fft=(fft_f, fft_a),
        envelope=(env_f, env_a),
        psd=(psd_f, psd_p),
        spectrogram=(spec_f, spec_t, spec_p),
        dominant_frequency=dominant,
        fft_comparison=fft_comparison,
        processed_signal=processed,
        normalization=normalize,
        detrend=detrend,
        geometry=geometry,
    )
