from __future__ import annotations

from typing import Dict, Tuple

import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt, resample


def load_audio(input_obj) -> Tuple[np.ndarray, int, Dict[str, float]]:
    signal, fs = librosa.load(input_obj, sr=None, mono=True)
    meta = {
        "source": "audio",
        "samples": float(len(signal)),
        "duration_sec": float(len(signal) / fs) if fs else 0.0,
    }
    return signal.astype(np.float32), fs, meta


def load_csv_signal(input_obj, value_column: str | None = None) -> Tuple[np.ndarray, Dict[str, float]]:
    df = pd.read_csv(input_obj)
    if df.empty:
        raise ValueError("CSV is empty.")

    if value_column is not None and value_column in df.columns:
        series = df[value_column]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("CSV must contain at least one numeric column.")
        series = df[numeric_cols[0]]
        value_column = numeric_cols[0]

    signal = series.astype(float).to_numpy()
    meta = {
        "source": "csv",
        "rows": float(len(df)),
        "value_column": value_column,
    }
    return signal.astype(np.float32), meta


def load_sensor_stream_csv(input_obj) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = pd.read_csv(input_obj)
    if df.empty:
        raise ValueError("Sensor stream CSV is empty.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Sensor stream CSV must include numeric columns.")

    meta = {
        "source": "sensor_stream",
        "rows": float(len(df)),
        "numeric_columns": float(len(numeric_cols)),
    }
    return df, meta


def resample_signal(signal: np.ndarray, src_fs: int, target_fs: int) -> np.ndarray:
    if src_fs == target_fs:
        return signal
    target_len = max(1, int(round(len(signal) * target_fs / src_fs)))
    return resample(signal, target_len).astype(np.float32)


def bandpass_filter(signal: np.ndarray, fs: int, lowcut: float = 20.0, highcut: float = 5000.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    highcut = min(highcut, nyq * 0.95)
    low = max(1e-6, lowcut / nyq)
    high = max(low + 1e-6, highcut / nyq)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def lowpass_filter(signal: np.ndarray, fs: int, cutoff: float = 1000.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    cutoff = min(cutoff, nyq * 0.95)
    normal_cutoff = max(1e-6, cutoff / nyq)
    b, a = butter(order, normal_cutoff, btype="low")
    return filtfilt(b, a, signal).astype(np.float32)


def detrend_signal(signal: np.ndarray) -> np.ndarray:
    return detrend(signal).astype(np.float32)


def normalize_signal(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    max_abs = float(np.max(np.abs(signal))) + eps
    return (signal / max_abs).astype(np.float32)


def preprocess_signal(signal: np.ndarray, fs: int, modality: str) -> Tuple[np.ndarray, Dict[str, float]]:
    cleaned = detrend_signal(signal)
    if modality == "audio":
        filtered = bandpass_filter(cleaned, fs, lowcut=20.0, highcut=5000.0)
    elif modality == "vibration":
        filtered = bandpass_filter(cleaned, fs, lowcut=5.0, highcut=min(5000.0, fs * 0.45))
    else:
        filtered = lowpass_filter(cleaned, fs, cutoff=min(100.0, fs * 0.4))

    normalized = normalize_signal(filtered)
    meta = {
        "modality": modality,
        "preprocess_mean": float(np.mean(normalized)),
        "preprocess_std": float(np.std(normalized)),
    }
    return normalized, meta
