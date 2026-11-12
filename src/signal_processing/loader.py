"""Load vibration signals from CSV / TXT and MATLAB ``.mat`` files."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.logger import get_logger

logger = get_logger("signal.loader")

MIN_SIGNAL_SAMPLES = 16


class SignalLoadError(ValueError):
    """Raised when a vibration file cannot be parsed or lacks valid numeric data."""


@dataclass
class LoadedSignal:
    """Container returned by :func:`load_vibration_file`."""

    signal: np.ndarray
    channel: str
    available_channels: list[str]
    file_format: str
    metadata: dict[str, Any]


def _is_number(value: object) -> bool:
    try:
        float(str(value))
        return True
    except ValueError:
        return False


def _read_csv_dataframe(source: str | Path | IO) -> pd.DataFrame:
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, sep=None, engine="python")
        # Headerless numeric file: first row parsed as numeric column headers.
        if len(df.columns) > 0 and all(_is_number(c) for c in df.columns):
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, sep=None, engine="python", header=None)
            df.columns = [f"ch{i}" for i in range(df.shape[1])]
    except Exception as exc:
        logger.error("Failed to read CSV/TXT signal file: %s", exc)
        raise SignalLoadError(f"Could not read CSV/TXT file: {exc}") from exc
    return df


def list_csv_channels(source: str | Path | IO) -> list[str]:
    """Return the numeric column names available in a CSV/TXT file."""
    df = _read_csv_dataframe(source)
    return [str(c) for c in df.select_dtypes(include="number").columns]


def load_csv_signal(source: str | Path | IO, column: str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Read a single vibration channel from a CSV/TXT file."""
    df = _read_csv_dataframe(source)
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise SignalLoadError("No numeric columns found in CSV/TXT file.")

    if column is None:
        column = str(numeric.columns[0])
    elif column not in numeric.columns:
        raise SignalLoadError(f"Column '{column}' not found or not numeric.")

    raw = numeric[column].dropna().to_numpy(dtype=float)
    if raw.size < MIN_SIGNAL_SAMPLES:
        raise SignalLoadError(
            f"Signal in column '{column}' is too short ({raw.size} samples; need at least {MIN_SIGNAL_SAMPLES})."
        )
    if not np.all(np.isfinite(raw)):
        raise SignalLoadError(f"Column '{column}' contains infinite values.")

    logger.info("Loaded %d CSV samples from column '%s'", raw.size, column)
    return raw, df


def _read_mat_dict(source: str | Path | IO) -> dict[str, Any]:
    try:
        if isinstance(source, (str, Path)):
            mat = loadmat(str(source), squeeze_me=False)
        else:
            if hasattr(source, "seek"):
                source.seek(0)
            raw_bytes = source.read()
            if isinstance(raw_bytes, str):
                raise SignalLoadError("MATLAB .mat input must be opened in binary mode.")
            mat = loadmat(io.BytesIO(raw_bytes), squeeze_me=False)
    except NotImplementedError as exc:
        logger.error("Unsupported MATLAB v7.3 HDF5 file: %s", exc)
        raise SignalLoadError(
            "MATLAB v7.3 (HDF5) .mat files are not supported by scipy.io.loadmat. "
            "Please re-save as MATLAB v7 (-v7) or export to CSV."
        ) from exc
    except SignalLoadError:
        raise
    except Exception as exc:
        logger.error("Failed to read MATLAB .mat file: %s", exc)
        raise SignalLoadError(f"Could not read MATLAB .mat file: {exc}") from exc
    return mat


def _extract_1d_numeric(arr: Any) -> np.ndarray | None:
    """Return a 1D float array if ``arr`` is a numeric vector of >= MIN_SIGNAL_SAMPLES."""
    if not isinstance(arr, np.ndarray):
        return None
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        return None
    squeezed = np.squeeze(arr)
    if squeezed.ndim == 1 and squeezed.size >= MIN_SIGNAL_SAMPLES:
        return np.asarray(squeezed, dtype=float)
    if squeezed.ndim == 2 and min(squeezed.shape) == 1 and squeezed.size >= MIN_SIGNAL_SAMPLES:
        return np.asarray(squeezed.ravel(), dtype=float)
    return None


def _extract_mat_signal_candidates(mat: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return usable 1D numeric signal vectors from a loaded .mat dictionary.

    Supports both flat MATLAB arrays (e.g. CWRU ``X097_DE_time``) and nested
    MATLAB structs (e.g. MFPT ``bearing.gs``).
    """
    candidates: dict[str, np.ndarray] = {}
    for key, val in mat.items():
        if key.startswith("__"):
            continue
        direct = _extract_1d_numeric(val)
        if direct is not None:
            candidates[key] = direct
            continue
        # Check for MATLAB struct arrays (structured dtype with named fields, e.g. MFPT `bearing`)
        if isinstance(val, np.ndarray) and val.dtype.names:
            for field_name in val.dtype.names:
                try:
                    field_val = val[field_name]
                    while isinstance(field_val, np.ndarray) and field_val.dtype == object and field_val.size == 1:
                        field_val = field_val.item()
                    vec = _extract_1d_numeric(field_val)
                    if vec is not None:
                        candidates[f"{key}.{field_name}"] = vec
                except Exception:
                    continue
    return candidates


def _extract_mat_scalars(mat: dict[str, Any], exclude_keys: set[str]) -> dict[str, float]:
    """Extract scalar numeric metadata from top-level and struct fields in a .mat dict."""
    scalars: dict[str, float] = {}
    for key, val in mat.items():
        if key.startswith("__") or key in exclude_keys:
            continue
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number) and val.size == 1:
            scalars[key] = float(val.item())
        elif isinstance(val, np.ndarray) and val.dtype.names:
            for field_name in val.dtype.names:
                full_key = f"{key}.{field_name}"
                if full_key in exclude_keys:
                    continue
                try:
                    field_val = val[field_name]
                    while isinstance(field_val, np.ndarray) and field_val.dtype == object and field_val.size == 1:
                        field_val = field_val.item()
                    if isinstance(field_val, np.ndarray) and np.issubdtype(field_val.dtype, np.number) and field_val.size == 1:
                        scalars[full_key] = float(field_val.item())
                    elif isinstance(field_val, (int, float)):
                        scalars[full_key] = float(field_val)
                except Exception:
                    continue
    return scalars


def list_mat_channels(source: str | Path | IO) -> list[str]:
    """Return variable names in a ``.mat`` file that hold valid 1D vibration arrays."""
    mat = _read_mat_dict(source)
    return list(_extract_mat_signal_candidates(mat).keys())


def load_mat_signal(
    source: str | Path | IO, variable: str | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a vibration signal vector from a MATLAB ``.mat`` file."""
    mat = _read_mat_dict(source)
    candidates = _extract_mat_signal_candidates(mat)
    if not candidates:
        raise SignalLoadError(
            f"No 1D numeric arrays with >= {MIN_SIGNAL_SAMPLES} samples found in .mat file."
        )

    if variable is None:
        # Prefer standard CWRU drive-end channel or MFPT bearing.gs if present, otherwise first candidate.
        preferred = next(
            (k for k in candidates if "DE_time" in k or k.endswith(".gs") or k == "gs"),
            None,
        )
        variable = preferred or next(iter(candidates))
    elif variable not in candidates:
        available = ", ".join(candidates.keys())
        raise SignalLoadError(f"Variable '{variable}' not found. Available signal variables: {available}")

    sig = candidates[variable]
    if not np.all(np.isfinite(sig)):
        raise SignalLoadError(f"MATLAB variable '{variable}' contains NaN or infinite values.")

    metadata: dict[str, Any] = {
        "selected_variable": variable,
        "available_variables": list(candidates.keys()),
    }
    metadata.update(_extract_mat_scalars(mat, set(candidates.keys())))

    logger.info("Loaded %d MATLAB samples from variable '%s'", sig.size, variable)
    return sig, metadata


def _infer_format(source: str | Path | IO, file_name: str | None = None) -> str:
    name = file_name
    if name is None and isinstance(source, (str, Path)):
        name = str(source)
    elif name is None and hasattr(source, "name"):
        name = str(getattr(source, "name"))

    if name:
        ext = Path(name).suffix.lower()
        if ext == ".mat":
            return "mat"
        if ext in {".csv", ".txt", ".tsv"}:
            return "csv"
    return "csv"


def load_vibration_file(
    source: str | Path | IO,
    file_name: str | None = None,
    channel: str | None = None,
) -> LoadedSignal:
    """Unified loader for CSV/TXT and MATLAB ``.mat`` vibration files."""
    fmt = _infer_format(source, file_name=file_name)
    if fmt == "mat":
        mat = _read_mat_dict(source)
        channels = list(_extract_mat_signal_candidates(mat).keys())
        sig, meta = load_mat_signal(source, variable=channel)
        selected = str(meta["selected_variable"])
        return LoadedSignal(
            signal=sig,
            channel=selected,
            available_channels=channels,
            file_format="mat",
            metadata=meta,
        )

    sig, df = load_csv_signal(source, column=channel)
    channels = [str(c) for c in df.select_dtypes(include="number").columns]
    selected = channel or channels[0]
    return LoadedSignal(
        signal=sig,
        channel=selected,
        available_channels=channels,
        file_format="csv",
        metadata={"columns": [str(c) for c in df.columns], "rows": int(len(df))},
    )


def load_signal(source: str | Path | IO, column: str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Backward-compatible CSV/TXT loader returning ``(signal, dataframe)``."""
    return load_csv_signal(source, column=column)
