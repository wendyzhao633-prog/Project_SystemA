"""Dataset-agnostic PPG feature extraction from MAX86140 CSV files.

This module only needs a CSV path, warmup duration, and sampling rate.
It has no dependency on RAVDESS-specific logic.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch


PPG_FEATURE_NAMES_TIME: list[str] = [
    "hr_mean_bpm",
    "hr_std_bpm",
    "rmssd_ms",
    "sdnn_ms",
    "rr_mean_ms",
    "rr_range_ms",
    "pnn50",
    "num_valid_peaks",
]
PPG_FEATURE_NAMES_FREQ: list[str] = [
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
]
PPG_FEATURE_NAMES_ALL: list[str] = PPG_FEATURE_NAMES_TIME + PPG_FEATURE_NAMES_FREQ

_MIN_RR_FOR_TIME: int = 5
_MIN_RR_FOR_FREQ: int = 10
_TRAPEZOID = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def extract_ppg_features(
    csv_path: Path,
    *,
    warmup_sec: float = 45.0,
    fs: int = 128,
    include_frequency_features: bool = True,
) -> dict[str, float]:
    """Extract HRV features from a MAX86140 PPG CSV file.

    Returns a dict whose keys are all names in ``PPG_FEATURE_NAMES_ALL``
    (or ``PPG_FEATURE_NAMES_TIME`` when ``include_frequency_features=False``).
    Features that cannot be computed due to insufficient data are 0.0.
    The returned dict never contains NaN.

    Args:
        csv_path: Path to the MAX86140 CSV recording.
        warmup_sec: Seconds of signal to discard from the start.
        fs: Sensor sampling rate in Hz.
        include_frequency_features: When True, also compute LF/HF power features.

    Returns:
        Dict mapping feature name → float value.
    """
    feature_names = PPG_FEATURE_NAMES_ALL if include_frequency_features else PPG_FEATURE_NAMES_TIME
    zero_result: dict[str, float] = {name: 0.0 for name in feature_names}

    signal, t = _parse_ppg_csv(csv_path, warmup_sec=warmup_sec)
    if signal is None or len(signal) < 2 * fs:
        return zero_result

    filtered = _bandpass_filter(signal, fs=fs)
    peak_times = _detect_peak_times(filtered, t, fs=fs)

    if len(peak_times) < 2:
        return zero_result

    rr_seconds = np.diff(peak_times)
    rr_valid = rr_seconds[(rr_seconds > 0.4) & (rr_seconds < 2.0)]

    features: dict[str, float] = {}
    if len(rr_valid) >= _MIN_RR_FOR_TIME:
        rr_ms = rr_valid * 1000.0
        successive_diff = np.diff(rr_ms)
        hr_bpm = 60000.0 / rr_ms
        pnn50 = (
            float(np.sum(np.abs(successive_diff) > 50.0) / len(successive_diff))
            if len(successive_diff) > 0
            else 0.0
        )
        features["hr_mean_bpm"] = float(np.mean(hr_bpm))
        features["hr_std_bpm"] = float(np.std(hr_bpm))
        features["rmssd_ms"] = (
            float(np.sqrt(np.mean(successive_diff**2)))
            if len(successive_diff) > 0
            else 0.0
        )
        features["sdnn_ms"] = float(np.std(rr_ms))
        features["rr_mean_ms"] = float(np.mean(rr_ms))
        features["rr_range_ms"] = float(np.max(rr_ms) - np.min(rr_ms))
        features["pnn50"] = pnn50
        features["num_valid_peaks"] = float(len(rr_valid) + 1)
    else:
        for name in PPG_FEATURE_NAMES_TIME:
            features[name] = 0.0

    if include_frequency_features:
        if len(rr_valid) >= _MIN_RR_FOR_FREQ:
            lf_power, hf_power, lf_hf_ratio = _compute_frequency_features(rr_valid)
        else:
            lf_power, hf_power, lf_hf_ratio = 0.0, 0.0, 0.0
        features["lf_power"] = lf_power
        features["hf_power"] = hf_power
        features["lf_hf_ratio"] = lf_hf_ratio

    return {name: float(0.0 if np.isnan(v) else v) for name, v in features.items()}


def features_to_array(
    features: dict[str, float],
    *,
    include_frequency_features: bool = True,
) -> np.ndarray:
    """Convert a feature dict to a fixed-order float32 numpy array.

    The order matches ``PPG_FEATURE_NAMES_ALL`` or ``PPG_FEATURE_NAMES_TIME``
    depending on ``include_frequency_features``.
    """
    names = PPG_FEATURE_NAMES_ALL if include_frequency_features else PPG_FEATURE_NAMES_TIME
    return np.array([features[name] for name in names], dtype=np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_ppg_csv(
    csv_path: Path,
    *,
    warmup_sec: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        text = Path(csv_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None

    lines = text.splitlines()
    header_idx: int | None = None
    stop_idx: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("timestamp") and header_idx is None:
            header_idx = i
        elif header_idx is not None and stripped.startswith("stop time"):
            stop_idx = i
            break

    if header_idx is None:
        return None, None

    data_end = stop_idx if stop_idx is not None else len(lines)
    csv_text = "\n".join(lines[header_idx:data_end])
    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception:
        return None, None

    if "timestamp" not in df.columns or "LEDC1" not in df.columns:
        return None, None

    ts = df["timestamp"].to_numpy(dtype=float)
    signal = df["LEDC1"].to_numpy(dtype=float)

    if np.nanmedian(ts) > 1e10:
        t = (ts - ts[0]) / 1000.0  # milliseconds → seconds
    else:
        t = ts - ts[0]

    valid_mask = t >= warmup_sec
    t = t[valid_mask]
    signal = signal[valid_mask]

    if len(signal) == 0:
        return None, None

    signal = np.where(np.isnan(signal), 0.0, signal)
    return signal, t


def _bandpass_filter(signal: np.ndarray, *, fs: int) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(3, [0.5 / nyq, 4.0 / nyq], btype="band")
    return filtfilt(b, a, signal)


def _detect_peak_times(
    filtered: np.ndarray,
    t: np.ndarray,
    *,
    fs: int,
) -> np.ndarray:
    std_val = float(np.std(filtered))
    prominence = max(0.3 * std_val, 1e-6)
    peaks, _ = find_peaks(filtered, distance=int(0.5 * fs), prominence=prominence)
    return t[peaks]


def _compute_frequency_features(rr_valid: np.ndarray) -> tuple[float, float, float]:
    """Compute LF/HF power via Welch PSD on an interpolated RR series at 4 Hz."""
    rr_ms = rr_valid * 1000.0
    cumulative_ms = np.cumsum(rr_ms)
    target_fs = 4.0
    t_interp = np.arange(0.0, cumulative_ms[-1], 1000.0 / target_fs)
    if len(t_interp) < 4:
        return 0.0, 0.0, 0.0

    rr_interp = np.interp(t_interp, cumulative_ms, rr_ms)
    freqs, psd = welch(rr_interp, fs=target_fs, nperseg=min(256, len(rr_interp)))

    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    lf_power = float(_TRAPEZOID(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
    hf_power = float(_TRAPEZOID(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
    lf_hf_ratio = lf_power / hf_power if hf_power > 1e-12 else 0.0
    return lf_power, hf_power, lf_hf_ratio
