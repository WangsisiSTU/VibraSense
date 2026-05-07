from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.stats import kurtosis, skew

from src.fft_tools import analyze_frequency


FREQUENCY_BANDS: Dict[str, Tuple[Tuple[float, float | None], ...]] = {
    "audio": ((0.0, 200.0), (200.0, 1000.0), (1000.0, None)),
    "vibration": ((0.0, 200.0), (200.0, 1000.0), (1000.0, None)),
    "default": ((0.0, 200.0), (200.0, 1000.0), (1000.0, None)),
}


def _safe_div(num: float, den: float) -> float:
    return float(num / (den + 1e-8))


def _select_band_ranges(modality: str) -> Iterable[Tuple[float, float | None]]:
    return FREQUENCY_BANDS.get(modality, FREQUENCY_BANDS["default"])


def _band_energy(freqs: np.ndarray, amps: np.ndarray, low: float, high: float | None) -> float:
    if len(freqs) == 0:
        return 0.0
    mask = freqs >= low
    if high is not None:
        mask &= freqs < high
    return float(np.sum(amps[mask]))


def extract_features(
    signal: np.ndarray,
    fs: int,
    modality: str,
    spectrum: Dict[str, object] | None = None,
) -> Dict[str, float]:
    signal = np.asarray(signal, dtype=float)
    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))
    crest = _safe_div(peak, rms)
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    kurt = float(kurtosis(signal, fisher=False, bias=False)) if len(signal) > 3 else 0.0
    skewness = float(skew(signal, bias=False)) if len(signal) > 3 else 0.0
    zcr = float(np.mean(np.diff(np.signbit(signal)) != 0)) if len(signal) > 1 else 0.0

    if spectrum is None:
        spectrum = analyze_frequency(signal, fs)
    freqs = np.asarray(spectrum.get("freqs", np.array([])), dtype=float)
    amps = np.asarray(spectrum.get("amps", np.array([])), dtype=float)

    spectral_energy = float(np.sum(amps**2))
    centroid = float(np.sum(freqs * amps) / (np.sum(amps) + 1e-8)) if len(freqs) else 0.0
    bandwidth = float(
        np.sqrt(np.sum(((freqs - centroid) ** 2) * amps) / (np.sum(amps) + 1e-8))
    ) if len(freqs) else 0.0

    band_ranges = tuple(_select_band_ranges(modality))
    low_band = _band_energy(freqs, amps, band_ranges[0][0], band_ranges[0][1])
    mid_band = _band_energy(freqs, amps, band_ranges[1][0], band_ranges[1][1])
    high_band = _band_energy(freqs, amps, band_ranges[2][0], band_ranges[2][1])
    total_band = low_band + mid_band + high_band + 1e-8

    return {
        "modality_audio": 1.0 if modality == "audio" else 0.0,
        "modality_vibration": 1.0 if modality == "vibration" else 0.0,
        "rms": rms,
        "peak": peak,
        "crest_factor": crest,
        "mean": mean_val,
        "std": std_val,
        "kurtosis": kurt,
        "skewness": skewness,
        "zcr": zcr,
        "spectral_energy": spectral_energy,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "energy_low_ratio": low_band / total_band,
        "energy_mid_ratio": mid_band / total_band,
        "energy_high_ratio": high_band / total_band,
    }
