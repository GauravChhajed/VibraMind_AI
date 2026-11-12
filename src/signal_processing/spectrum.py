"""Frequency-domain and time-frequency analysis: FFT, PSD, STFT, spectrogram, envelope."""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from src.signal_processing.preprocessing import validate_sampling_rate, validate_signal


def _adapt_segment_params(
    n_samples: int, nperseg: int | None = None, noverlap: int | None = None
) -> tuple[int, int]:
    """Choose valid ``(nperseg, noverlap)`` for any signal length >= 4."""
    if n_samples < 4:
        raise ValueError(f"Signal must contain at least 4 samples for windowed spectral analysis (got {n_samples}).")
    default_seg = min(512, max(16, n_samples // 8))
    seg = min(int(nperseg) if nperseg is not None else default_seg, n_samples)
    seg = max(4, seg)
    ovl = int(noverlap) if noverlap is not None else seg // 2
    ovl = max(0, min(ovl, seg - 1))
    return seg, ovl


def compute_fft(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum using a Hann window."""
    fs_val = validate_sampling_rate(fs)
    x = validate_signal(signal, min_samples=2)
    x = x - float(np.mean(x))
    window = np.hanning(x.size)
    win_sum = float(window.sum()) or float(x.size)
    spectrum = np.fft.rfft(x * window)
    amplitude = 2.0 * np.abs(spectrum) / win_sum
    # DC and Nyquist bins (if even length) are not doubled in single-sided scaling.
    amplitude[0] *= 0.5
    if x.size % 2 == 0 and amplitude.size > 1:
        amplitude[-1] *= 0.5
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs_val)
    return freqs, amplitude


def extract_dominant_frequency(freqs: np.ndarray, amplitude: np.ndarray) -> dict[str, float]:
    """Return the dominant non-DC frequency and its amplitude from an FFT spectrum."""
    f = np.asarray(freqs, dtype=float)
    a = np.asarray(amplitude, dtype=float)
    if f.ndim != 1 or a.ndim != 1 or f.size != a.size or f.size < 2:
        raise ValueError("freqs and amplitude must be 1D arrays of equal length >= 2.")
    start = 1 if f[0] == 0.0 else 0
    idx = start + int(np.argmax(a[start:]))
    return {"frequency_hz": float(f[idx]), "amplitude": float(a[idx])}


def extract_significant_peaks(
    freqs: np.ndarray,
    amplitude: np.ndarray,
    *,
    max_peaks: int = 15,
    min_prominence_ratio: float = 0.05,
    min_snr_db: float = 6.0,
    min_distance_hz: float = 2.0,
) -> list[dict[str, float]]:
    """Extract significant observed spectral peaks from an FFT spectrum.

    Filters peaks by relative prominence against the max amplitude and SNR above
    the global median spectral noise floor, skipping the 0 Hz DC bin.
    """
    f = np.asarray(freqs, dtype=float)
    a = np.asarray(amplitude, dtype=float)
    if f.ndim != 1 or a.ndim != 1 or f.size != a.size or f.size < 3:
        raise ValueError("freqs and amplitude must be 1D arrays of equal length >= 3.")

    df = float(f[1] - f[0]) if f.size > 1 else 1.0
    distance_bins = max(1, int(round(min_distance_hz / max(df, 1e-12))))
    max_amp = float(np.max(a[1:])) if a.size > 1 else float(np.max(a))
    if max_amp <= 0.0:
        return []

    noise_floor = float(np.median(a[1:])) if a.size > 1 else float(np.median(a))
    prominence_thresh = max(max_amp * min_prominence_ratio, noise_floor * 1.5)

    peak_indices, props = sps.find_peaks(a, prominence=prominence_thresh, distance=distance_bins)
    peaks: list[dict[str, float]] = []
    for idx, prom in zip(peak_indices, props["prominences"]):
        if idx == 0 and f[0] == 0.0:
            continue
        amp = float(a[idx])
        snr_db = float(20.0 * np.log10((amp + 1e-12) / (noise_floor + 1e-12)))
        if snr_db < min_snr_db:
            continue
        peaks.append(
            {
                "frequency_hz": float(f[idx]),
                "amplitude": amp,
                "prominence": float(prom),
                "snr_db": snr_db,
            }
        )

    peaks.sort(key=lambda p: p["amplitude"], reverse=True)
    return peaks[:max_peaks]


def compute_psd(
    signal: np.ndarray,
    fs: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch Power Spectral Density (PSD) in units^2/Hz."""
    fs_val = validate_sampling_rate(fs)
    x = validate_signal(signal, min_samples=4)
    x = x - float(np.mean(x))
    seg, ovl = _adapt_segment_params(x.size, nperseg=nperseg, noverlap=noverlap)
    freqs, psd = sps.welch(x, fs=fs_val, window="hann", nperseg=seg, noverlap=ovl, detrend="constant")
    return freqs, psd


def compute_stft(
    signal: np.ndarray,
    fs: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Short-Time Fourier Transform (STFT).

    Returns ``(freqs, times, zxx_complex)``.
    """
    fs_val = validate_sampling_rate(fs)
    x = validate_signal(signal, min_samples=4)
    x = x - float(np.mean(x))
    seg, ovl = _adapt_segment_params(x.size, nperseg=nperseg, noverlap=noverlap)
    freqs, times, zxx = sps.stft(x, fs=fs_val, window="hann", nperseg=seg, noverlap=ovl, boundary="zeros")
    return freqs, times, zxx


def compute_spectrogram(
    signal: np.ndarray,
    fs: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute power spectrogram from STFT.

    Returns ``(freqs, times, power_matrix)`` with shape ``(len(freqs), len(times))``.
    """
    freqs, times, zxx = compute_stft(signal, fs, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(zxx) ** 2
    return freqs, times, power


def bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    fs_val = validate_sampling_rate(fs)
    nyq = fs_val / 2.0
    if not 0 < low < high < nyq:
        raise ValueError(f"Band must satisfy 0 < low < high < Nyquist ({nyq:.1f} Hz).")
    x = validate_signal(signal, min_samples=16)
    sos = sps.butter(order, [low, high], btype="bandpass", fs=fs_val, output="sos")
    return sps.sosfiltfilt(sos, x)


def envelope_spectrum(
    signal: np.ndarray, fs: float, band: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Envelope (demodulated) spectrum - key tool for bearing defect detection."""
    fs_val = validate_sampling_rate(fs)
    x = validate_signal(signal, min_samples=16)
    if band is not None:
        x = bandpass_filter(x, fs_val, *band)
    envelope = np.abs(sps.hilbert(x))
    return compute_fft(envelope, fs_val)


def peak_near(
    freqs: np.ndarray, amplitude: np.ndarray, target: float, tolerance: float = 0.03
) -> tuple[float, float]:
    """Return ``(frequency, amplitude)`` of the largest peak within ``±tolerance * target``."""
    width = max(target * tolerance, freqs[1] - freqs[0] if freqs.size > 1 else 0.0)
    mask = (freqs >= target - width) & (freqs <= target + width)
    if not mask.any():
        return target, 0.0
    idx = int(np.argmax(amplitude[mask]))
    return float(freqs[mask][idx]), float(amplitude[mask][idx])
