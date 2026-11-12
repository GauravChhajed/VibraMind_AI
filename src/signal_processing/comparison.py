"""Compare observed FFT peaks against theoretical bearing characteristic frequencies.

Theoretical frequencies (FTF, BPFO, BPFI, BSF) are kinematic references calculated
from shaft speed and bearing geometry. Observed peaks are empirical spectral peaks
measured from the vibration FFT. Matching a frequency within tolerance is a screening
indicator and NEVER definitive proof of a bearing fault on its own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.signal_processing.spectrum import extract_significant_peaks

FAULT_TYPES = ("FTF", "BPFO", "BPFI", "BSF")
FAULT_DESCRIPTIONS = {
    "FTF": "Fundamental Train (Cage) Frequency",
    "BPFO": "Ball Pass Frequency Outer Race",
    "BPFI": "Ball Pass Frequency Inner Race",
    "BSF": "Ball Spin Frequency",
}
NON_CONCLUSIVE_DISCLAIMER = (
    "Theoretical frequencies are kinematic estimates assuming no slip. "
    "Observed peak matches within tolerance are screening indicators only and "
    "do NOT constitute definitive proof of a bearing fault; confirm with envelope "
    "analysis, time-domain indicators, harmonics/sidebands, and machine context."
)


@dataclass(frozen=True)
class HarmonicMatch:
    harmonic_order: int
    theoretical_harmonic_hz: float
    within_nyquist: bool
    matched_observed_hz: float | None
    matched_amplitude: float | None
    matched_snr_db: float | None
    frequency_diff_hz: float | None
    relative_diff_pct: float | None
    matched: bool


@dataclass(frozen=True)
class FaultFrequencyComparison:
    fault_type: str
    description: str
    theoretical_hz: float
    harmonics_checked: list[HarmonicMatch]
    matched_harmonic_count: int
    possible_match: bool
    note: str


@dataclass(frozen=True)
class FaultComparisonReport:
    observed_peaks: list[dict[str, float]]
    theoretical_frequencies: dict[str, float]
    tolerance_pct: float
    min_tolerance_hz: float
    comparisons: dict[str, FaultFrequencyComparison]
    possible_matches: list[str]
    disclaimer: str = NON_CONCLUSIVE_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain JSON-compatible dictionaries for LLM tool use or UI display."""
        return {
            "observed_peaks": list(self.observed_peaks),
            "theoretical_frequencies": dict(self.theoretical_frequencies),
            "tolerance_pct": self.tolerance_pct,
            "min_tolerance_hz": self.min_tolerance_hz,
            "comparisons": {k: asdict(v) for k, v in self.comparisons.items()},
            "possible_matches": list(self.possible_matches),
            "disclaimer": self.disclaimer,
        }


def compare_fault_frequencies(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    theoretical_frequencies: dict[str, float],
    *,
    max_harmonics: int = 3,
    tolerance_pct: float = 3.0,
    min_tolerance_hz: float | None = None,
    max_peaks: int = 20,
    min_prominence_ratio: float = 0.05,
    min_snr_db: float = 6.0,
) -> FaultComparisonReport:
    """Extract significant observed FFT peaks and compare them against theoretical bearing frequencies.

    Parameters
    ----------
    freqs, amplitudes:
        Single-sided FFT frequency axis (Hz) and amplitude array.
    theoretical_frequencies:
        Dictionary containing ``FTF``, ``BPFO``, ``BPFI``, and ``BSF`` in Hz.
    max_harmonics:
        Number of harmonics (1x, 2x, ..., ``max_harmonics``x) to inspect per fault type.
    tolerance_pct:
        Relative matching tolerance around each theoretical harmonic (%).
    min_tolerance_hz:
        Floor on matching tolerance in Hz (defaults to ``max(1.0, 1.5 * df)``).
    """
    f = np.asarray(freqs, dtype=float)
    a = np.asarray(amplitudes, dtype=float)
    if f.ndim != 1 or a.ndim != 1 or f.size != a.size or f.size < 3:
        raise ValueError("freqs and amplitudes must be 1D arrays of equal length >= 3.")
    if max_harmonics < 1:
        raise ValueError("max_harmonics must be >= 1.")
    if tolerance_pct <= 0:
        raise ValueError("tolerance_pct must be positive.")

    missing = [k for k in FAULT_TYPES if k not in theoretical_frequencies]
    if missing:
        raise ValueError(f"theoretical_frequencies is missing required keys: {', '.join(missing)}")

    df = float(f[1] - f[0])
    nyquist = float(f[-1])
    floor_tol = float(min_tolerance_hz) if min_tolerance_hz is not None else max(1.0, 1.5 * df)

    observed_peaks = extract_significant_peaks(
        f,
        a,
        max_peaks=max_peaks,
        min_prominence_ratio=min_prominence_ratio,
        min_snr_db=min_snr_db,
        min_distance_hz=max(1.0, 2.0 * df),
    )

    comparisons: dict[str, FaultFrequencyComparison] = {}
    possible_matches: list[str] = []

    for fault_type in FAULT_TYPES:
        base_hz = float(theoretical_frequencies[fault_type])
        if base_hz <= 0 or not np.isfinite(base_hz):
            raise ValueError(f"Theoretical frequency '{fault_type}' must be positive (got {base_hz}).")

        harmonic_matches: list[HarmonicMatch] = []
        for order in range(1, max_harmonics + 1):
            target_hz = base_hz * order
            if target_hz > nyquist:
                harmonic_matches.append(
                    HarmonicMatch(
                        harmonic_order=order,
                        theoretical_harmonic_hz=target_hz,
                        within_nyquist=False,
                        matched_observed_hz=None,
                        matched_amplitude=None,
                        matched_snr_db=None,
                        frequency_diff_hz=None,
                        relative_diff_pct=None,
                        matched=False,
                    )
                )
                continue

            tol_hz = max(floor_tol, target_hz * (tolerance_pct / 100.0))
            candidates = [
                p for p in observed_peaks if abs(p["frequency_hz"] - target_hz) <= tol_hz
            ]
            if candidates:
                # Pick closest observed peak in frequency (tie-break on highest amplitude).
                best = min(
                    candidates,
                    key=lambda p: (abs(p["frequency_hz"] - target_hz), -p["amplitude"]),
                )
                diff_hz = best["frequency_hz"] - target_hz
                rel_pct = 100.0 * abs(diff_hz) / target_hz
                harmonic_matches.append(
                    HarmonicMatch(
                        harmonic_order=order,
                        theoretical_harmonic_hz=target_hz,
                        within_nyquist=True,
                        matched_observed_hz=best["frequency_hz"],
                        matched_amplitude=best["amplitude"],
                        matched_snr_db=best["snr_db"],
                        frequency_diff_hz=float(diff_hz),
                        relative_diff_pct=float(rel_pct),
                        matched=True,
                    )
                )
            else:
                # Report nearest observed peak distance if any peak exists, with matched=False.
                nearest = (
                    min(observed_peaks, key=lambda p: abs(p["frequency_hz"] - target_hz))
                    if observed_peaks
                    else None
                )
                diff_hz = (nearest["frequency_hz"] - target_hz) if nearest else None
                rel_pct = (100.0 * abs(diff_hz) / target_hz) if diff_hz is not None else None
                harmonic_matches.append(
                    HarmonicMatch(
                        harmonic_order=order,
                        theoretical_harmonic_hz=target_hz,
                        within_nyquist=True,
                        matched_observed_hz=nearest["frequency_hz"] if nearest else None,
                        matched_amplitude=nearest["amplitude"] if nearest else None,
                        matched_snr_db=nearest["snr_db"] if nearest else None,
                        frequency_diff_hz=float(diff_hz) if diff_hz is not None else None,
                        relative_diff_pct=float(rel_pct) if rel_pct is not None else None,
                        matched=False,
                    )
                )

        matched_orders = [h.harmonic_order for h in harmonic_matches if h.matched]
        possible = len(matched_orders) > 0
        if possible:
            possible_matches.append(fault_type)
            orders_str = ", ".join(f"{o}x" for o in matched_orders)
            note = (
                f"Possible frequency alignment at {orders_str} of theoretical {fault_type} "
                f"({base_hz:.2f} Hz). Treat as a screening indicator, not definitive proof."
            )
        else:
            note = f"No significant observed FFT peak matched theoretical {fault_type} ({base_hz:.2f} Hz) within tolerance."

        comparisons[fault_type] = FaultFrequencyComparison(
            fault_type=fault_type,
            description=FAULT_DESCRIPTIONS[fault_type],
            theoretical_hz=base_hz,
            harmonics_checked=harmonic_matches,
            matched_harmonic_count=len(matched_orders),
            possible_match=possible,
            note=note,
        )

    return FaultComparisonReport(
        observed_peaks=observed_peaks,
        theoretical_frequencies={k: float(theoretical_frequencies[k]) for k in FAULT_TYPES},
        tolerance_pct=float(tolerance_pct),
        min_tolerance_hz=floor_tol,
        comparisons=comparisons,
        possible_matches=possible_matches,
    )

