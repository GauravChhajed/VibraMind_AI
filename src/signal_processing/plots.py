"""Modular Plotly visualizations for bearing vibration analysis."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

FAULT_COLORS = {
    "FTF": "#8e44ad",
    "BPFO": "#e74c3c",
    "BPFI": "#e67e22",
    "BSF": "#27ae60",
}


def plot_waveform(
    signal: np.ndarray,
    fs: float,
    *,
    title: str = "Original Vibration Waveform",
    max_seconds: float | None = None,
    yaxis_title: str = "Amplitude",
) -> go.Figure:
    """Plot time-domain vibration waveform."""
    x = np.asarray(signal, dtype=float)
    n = x.size if max_seconds is None else min(x.size, max(1, int(fs * max_seconds)))
    t = np.arange(n) / float(fs)
    fig = go.Figure(
        go.Scatter(x=t, y=x[:n], mode="lines", line=dict(width=1), name="Waveform")
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=yaxis_title,
        height=320,
        margin=dict(l=50, r=20, t=45, b=40),
    )
    return fig


def _add_theoretical_markers(
    fig: go.Figure,
    theoretical_frequencies: dict[str, float] | None,
    *,
    max_freq: float,
) -> None:
    if not theoretical_frequencies:
        return
    for name in ("FTF", "BPFO", "BPFI", "BSF"):
        freq = theoretical_frequencies.get(name)
        if freq is None or not (0 < freq <= max_freq):
            continue
        color = FAULT_COLORS.get(name, "#7f8c8d")
        fig.add_vline(
            x=freq,
            line_dash="dash",
            line_color=color,
            line_width=1.5,
            annotation_text=f"Theoretical {name} ({freq:.1f} Hz)",
            annotation_position="top",
        )


def plot_fft_spectrum(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    *,
    theoretical_frequencies: dict[str, float] | None = None,
    observed_peaks: list[dict[str, float]] | None = None,
    max_freq: float | None = None,
    title: str = "FFT Frequency Spectrum (Observed Peaks vs Theoretical Frequencies)",
) -> go.Figure:
    """Plot single-sided FFT amplitude spectrum with theoretical fault markers."""
    f = np.asarray(freqs, dtype=float)
    a = np.asarray(amplitudes, dtype=float)
    limit = float(max_freq) if max_freq is not None else float(f[-1])
    mask = f <= limit

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=f[mask],
            y=a[mask],
            mode="lines",
            line=dict(width=1.2, color="#1f77b4"),
            name="Observed FFT Spectrum",
        )
    )

    if observed_peaks:
        vis_peaks = [p for p in observed_peaks if p["frequency_hz"] <= limit]
        if vis_peaks:
            fig.add_trace(
                go.Scatter(
                    x=[p["frequency_hz"] for p in vis_peaks],
                    y=[p["amplitude"] for p in vis_peaks],
                    mode="markers",
                    marker=dict(size=8, color="#d62728", symbol="diamond"),
                    name="Observed Significant Peaks",
                    hovertemplate="Observed Peak: %{x:.2f} Hz<br>Amplitude: %{y:.4f}<extra></extra>",
                )
            )

    _add_theoretical_markers(fig, theoretical_frequencies, max_freq=limit)
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=65, b=40),
    )
    return fig


def plot_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    *,
    theoretical_frequencies: dict[str, float] | None = None,
    max_freq: float | None = None,
    log_y: bool = True,
    title: str = "Welch Power Spectral Density (PSD)",
) -> go.Figure:
    """Plot Welch PSD with optional theoretical bearing frequency lines."""
    f = np.asarray(freqs, dtype=float)
    p = np.asarray(psd, dtype=float)
    limit = float(max_freq) if max_freq is not None else float(f[-1])
    mask = f <= limit

    fig = go.Figure(
        go.Scatter(
            x=f[mask],
            y=p[mask],
            mode="lines",
            line=dict(width=1.3, color="#2c3e50"),
            name="Welch PSD",
        )
    )
    _add_theoretical_markers(fig, theoretical_frequencies, max_freq=limit)
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="PSD (units²/Hz)",
        yaxis_type="log" if log_y else "linear",
        height=360,
        margin=dict(l=50, r=20, t=45, b=40),
    )
    return fig


def plot_spectrogram(
    freqs: np.ndarray,
    times: np.ndarray,
    power: np.ndarray,
    *,
    max_freq: float | None = None,
    title: str = "STFT Spectrogram (dB)",
) -> go.Figure:
    """Plot time-frequency spectrogram heatmap in decibels."""
    f = np.asarray(freqs, dtype=float)
    t = np.asarray(times, dtype=float)
    sxx = np.asarray(power, dtype=float)
    limit = float(max_freq) if max_freq is not None else float(f[-1])
    mask = f <= limit

    power_db = 10.0 * np.log10(np.maximum(sxx[mask, :], 1e-12))
    fig = go.Figure(
        go.Heatmap(
            x=t,
            y=f[mask],
            z=power_db,
            colorscale="Viridis",
            colorbar=dict(title="Power (dB)"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=380,
        margin=dict(l=50, r=20, t=45, b=40),
    )
    return fig


def plot_envelope_spectrum(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    *,
    theoretical_frequencies: dict[str, float] | None = None,
    max_freq: float | None = None,
    title: str = "Envelope Spectrum",
) -> go.Figure:
    """Plot demodulated envelope spectrum with theoretical fault frequencies."""
    f = np.asarray(freqs, dtype=float)
    a = np.asarray(amplitudes, dtype=float)
    limit = float(max_freq) if max_freq is not None else float(f[-1])
    mask = f <= limit

    fig = go.Figure(
        go.Scatter(
            x=f[mask],
            y=a[mask],
            mode="lines",
            line=dict(width=1.2, color="#16a085"),
            name="Envelope",
        )
    )
    _add_theoretical_markers(fig, theoretical_frequencies, max_freq=limit)
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        height=360,
        margin=dict(l=50, r=20, t=45, b=40),
    )
    return fig

