"""Synthetic bearing vibration signals for demos and tests (no real data needed)."""

from __future__ import annotations

import numpy as np


def generate_bearing_signal(
    fs: float = 12_000,
    duration: float = 1.0,
    shaft_hz: float = 29.95,
    fault_freq: float | None = None,
    resonance_hz: float = 3_000,
    impulse_amplitude: float = 1.5,
    noise_std: float = 0.3,
    seed: int | None = 42,
) -> np.ndarray:
    """Simulate an accelerometer signal.

    The model combines:
    - Shaft rotation tone (1x shaft_hz) + white Gaussian noise.
    - When ``fault_freq`` is provided:
      1. Baseband fault harmonics (1x, 2x, 3x ``fault_freq``) visible directly
         in the FFT and Welch PSD spectra.
      2. A train of exponentially decaying structural resonance impulses at
         ``resonance_hz`` repeating at ``fault_freq`` (visible in the envelope
         spectrum and spectrogram).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration)) / fs
    x = 0.2 * np.sin(2 * np.pi * shaft_hz * t) + rng.normal(0, noise_std, t.size)

    if fault_freq:
        # Baseband fault harmonics (1x, 2x, 3x) within Nyquist
        nyq = fs / 2.0
        for order, scale in ((1, 0.45), (2, 0.25), (3, 0.12)):
            harm_hz = fault_freq * order
            if harm_hz < nyq:
                x += (impulse_amplitude * scale) * np.sin(2 * np.pi * harm_hz * t)

        # High-frequency resonance impulse train
        decay = 800.0  # 1/s
        ring_len = max(4, int(fs * 0.01))
        tau = np.arange(ring_len) / fs
        ring = np.exp(-decay * tau) * np.sin(2 * np.pi * resonance_hz * tau)
        impulses = np.zeros_like(t)
        period = fs / fault_freq
        for k in np.arange(0, t.size, period):
            # Small random jitter models rolling-element slip.
            idx = int(k + rng.normal(0, 0.01 * period))
            if 0 <= idx < t.size:
                impulses[idx] = impulse_amplitude
        x += np.convolve(impulses, ring)[: t.size]
    return x
