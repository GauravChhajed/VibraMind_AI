"""Signal-processing utilities for bearing vibration analysis (NumPy/SciPy)."""

from src.signal_processing.bearing_frequencies import (
    CWRU_SKF_6205_2RS,
    MFPT_NICE_BEARING,
    BearingGeometry,
    bearing_fault_frequencies,
    rpm_to_hz,
)
from src.signal_processing.comparison import (
    FaultComparisonReport,
    FaultFrequencyComparison,
    HarmonicMatch,
    compare_fault_frequencies,
)
from src.signal_processing.features import time_domain_features
from src.signal_processing.loader import (
    LoadedSignal,
    SignalLoadError,
    list_csv_channels,
    list_mat_channels,
    load_csv_signal,
    load_mat_signal,
    load_signal,
    load_vibration_file,
)
from src.signal_processing.pipeline import SignalAnalysisBundle, run_signal_processing_pipeline
from src.signal_processing.plots import (
    plot_envelope_spectrum,
    plot_fft_spectrum,
    plot_psd,
    plot_spectrogram,
    plot_waveform,
)
from src.signal_processing.preprocessing import (
    VALID_DETREND,
    VALID_NORMALIZATIONS,
    normalize_signal,
    preprocess_signal,
    validate_sampling_rate,
    validate_signal,
)
from src.signal_processing.spectrum import (
    bandpass_filter,
    compute_fft,
    compute_psd,
    compute_spectrogram,
    compute_stft,
    envelope_spectrum,
    extract_dominant_frequency,
    extract_significant_peaks,
    peak_near,
)
from src.signal_processing.synthetic import generate_bearing_signal

__all__ = [
    "BearingGeometry",
    "CWRU_SKF_6205_2RS",
    "MFPT_NICE_BEARING",
    "bearing_fault_frequencies",
    "rpm_to_hz",
    "FaultComparisonReport",
    "FaultFrequencyComparison",
    "HarmonicMatch",
    "compare_fault_frequencies",
    "time_domain_features",
    "LoadedSignal",
    "SignalLoadError",
    "list_csv_channels",
    "list_mat_channels",
    "load_csv_signal",
    "load_mat_signal",
    "load_signal",
    "load_vibration_file",
    "SignalAnalysisBundle",
    "run_signal_processing_pipeline",
    "plot_envelope_spectrum",
    "plot_fft_spectrum",
    "plot_psd",
    "plot_spectrogram",
    "plot_waveform",
    "VALID_DETREND",
    "VALID_NORMALIZATIONS",
    "normalize_signal",
    "preprocess_signal",
    "validate_sampling_rate",
    "validate_signal",
    "bandpass_filter",
    "compute_fft",
    "compute_psd",
    "compute_spectrogram",
    "compute_stft",
    "envelope_spectrum",
    "extract_dominant_frequency",
    "extract_significant_peaks",
    "peak_near",
    "generate_bearing_signal",
]
