"""Time-domain condition indicators."""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.signal_processing.preprocessing import validate_signal

# Re-exported for internal modules expecting `_validate`.
_validate = validate_signal


def time_domain_features(signal: np.ndarray) -> dict[str, float]:
    """Compute time-domain vibration condition indicators.

    Returns mean, variance, standard deviation, RMS, kurtosis (Pearson,
    Gaussian = 3), crest factor, peak, peak-to-peak, skewness, shape factor,
    and impulse factor.
    """
    x = validate_signal(signal, min_samples=2)
    abs_x = np.abs(x)
    mean = float(np.mean(x))
    variance = float(np.var(x))
    std = float(np.sqrt(variance))
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(abs_x.max())
    mean_abs = float(abs_x.mean())

    if std == 0.0:
        kurtosis_val = 0.0
        skewness_val = 0.0
    else:
        kurtosis_val = float(stats.kurtosis(x, fisher=False))
        skewness_val = float(stats.skew(x))

    crest_factor = peak / rms if rms > 0.0 else 0.0
    shape_factor = rms / mean_abs if mean_abs > 0.0 else 0.0
    impulse_factor = peak / mean_abs if mean_abs > 0.0 else 0.0

    return {
        "mean": mean,
        "variance": variance,
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(x.max() - x.min()),
        "crest_factor": float(crest_factor),
        "kurtosis": kurtosis_val,
        "skewness": skewness_val,
        "shape_factor": float(shape_factor),
        "impulse_factor": float(impulse_factor),
    }
