"""Signal validation, preprocessing, and normalization utilities."""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from src.logger import get_logger

logger = get_logger("signal.preprocessing")

VALID_NORMALIZATIONS = ("none", "zscore", "minmax", "peak", "rms")
VALID_DETREND = ("none", "constant", "linear")


def validate_signal(signal: np.ndarray, min_samples: int = 2) -> np.ndarray:
    """Validate and coerce ``signal`` to a finite 1D float64 array."""
    arr = np.asarray(signal, dtype=float)
    if arr.ndim > 1:
        if min(arr.shape) == 1:
            arr = arr.ravel()
        else:
            raise ValueError(f"Expected a 1D vibration signal, got shape {arr.shape}.")
    else:
        arr = arr.ravel()

    if arr.size < min_samples:
        raise ValueError(f"Signal must contain at least {min_samples} samples (got {arr.size}).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Signal contains NaN or infinite values.")
    return arr


def validate_sampling_rate(fs: float | None) -> float:
    """Validate that sampling frequency ``fs`` (Hz) is explicitly provided and positive."""
    if fs is None:
        raise ValueError("Sampling frequency (fs) must be explicitly provided and cannot be assumed.")
    try:
        fs_val = float(fs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sampling frequency must be a positive number, got {fs!r}.") from exc
    if not np.isfinite(fs_val) or fs_val <= 0:
        raise ValueError(f"Sampling frequency must be a finite positive value in Hz (got {fs_val}).")
    return fs_val


def normalize_signal(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize a 1D vibration signal.

    Supported methods:
    - ``"none"``: return a float64 copy unchanged.
    - ``"zscore"``: zero mean, unit standard deviation.
    - ``"minmax"``: scale to ``[-1, 1]``.
    - ``"peak"``: divide by maximum absolute amplitude.
    - ``"rms"``: divide by root-mean-square amplitude.
    """
    x = validate_signal(signal)
    key = method.strip().lower()
    if key not in VALID_NORMALIZATIONS:
        raise ValueError(f"Unknown normalization method '{method}'. Choose from {VALID_NORMALIZATIONS}.")

    if key == "none":
        return x.copy()

    if key == "zscore":
        std = float(np.std(x))
        if std == 0.0:
            raise ValueError("Cannot z-score normalize a constant signal (standard deviation is zero).")
        return (x - float(np.mean(x))) / std

    if key == "minmax":
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        span = x_max - x_min
        if span == 0.0:
            raise ValueError("Cannot min-max normalize a constant signal (min equals max).")
        return 2.0 * (x - x_min) / span - 1.0

    if key == "peak":
        peak = float(np.max(np.abs(x)))
        if peak == 0.0:
            raise ValueError("Cannot peak-normalize an all-zero signal.")
        return x / peak

    # key == "rms"
    rms = float(np.sqrt(np.mean(x**2)))
    if rms == 0.0:
        raise ValueError("Cannot RMS-normalize an all-zero signal.")
    return x / rms


def preprocess_signal(
    signal: np.ndarray,
    fs: float,
    detrend: str = "constant",
    normalize: str = "none",
    bandpass: tuple[float, float] | None = None,
    filter_order: int = 4,
) -> np.ndarray:
    """Validate, detrend, optionally band-pass filter, and normalize a signal."""
    fs_val = validate_sampling_rate(fs)
    x = validate_signal(signal)

    detrend_key = detrend.strip().lower()
    if detrend_key not in VALID_DETREND:
        raise ValueError(f"Unknown detrend mode '{detrend}'. Choose from {VALID_DETREND}.")

    if detrend_key == "constant":
        x = x - float(np.mean(x))
    elif detrend_key == "linear":
        x = np.asarray(sps.detrend(x, type="linear"), dtype=float)

    if bandpass is not None:
        if len(bandpass) != 2:
            raise ValueError("bandpass must be a (low_hz, high_hz) tuple.")
        low, high = float(bandpass[0]), float(bandpass[1])
        nyq = fs_val / 2.0
        if not (0.0 < low < high < nyq):
            raise ValueError(f"Band-pass limits must satisfy 0 < low < high < Nyquist ({nyq:.2f} Hz).")
        sos = sps.butter(filter_order, [low, high], btype="bandpass", fs=fs_val, output="sos")
        # sosfiltfilt requires padlen < signal length (default padlen = 3*(2*n_sections + 1 - min(...)))
        if x.size <= 3 * (2 * sos.shape[0] + 1):
            raise ValueError(
                f"Signal length ({x.size} samples) is too short for zero-phase order-{filter_order} filtering."
            )
        x = np.asarray(sps.sosfiltfilt(sos, x), dtype=float)

    return normalize_signal(x, method=normalize)

