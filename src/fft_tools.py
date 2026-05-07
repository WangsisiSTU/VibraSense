from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import find_peaks


def compute_fft(signal: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([])
    fft_vals = np.fft.rfft(signal)
    amps = np.abs(fft_vals) / max(1, n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, amps


def detect_main_peaks(freqs: np.ndarray, amps: np.ndarray, top_k: int = 3, min_distance_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    if len(freqs) == 0:
        return np.array([]), np.array([])

    peaks, _ = find_peaks(amps, distance=min_distance_bins)
    if len(peaks) == 0:
        peak_idx = int(np.argmax(amps))
        return np.array([freqs[peak_idx]]), np.array([amps[peak_idx]])

    peak_amps = amps[peaks]
    idx = np.argsort(peak_amps)[-top_k:]
    top_freqs = freqs[peaks][idx]
    top_amps = peak_amps[idx]
    order = np.argsort(top_amps)[::-1]
    return top_freqs[order], top_amps[order]


def harmonic_candidates(main_freq: float, max_harmonic: int = 5) -> List[float]:
    if main_freq <= 0:
        return []
    return [main_freq * i for i in range(2, max_harmonic + 1)]


def peak_confidence(amps: np.ndarray, top_amps: np.ndarray) -> float:
    if len(amps) == 0 or len(top_amps) == 0:
        return 0.0
    total = float(np.sum(amps)) + 1e-8
    top = float(np.sum(top_amps))
    return float(np.clip(top / total, 0.0, 1.0))


def summarize_spectrum(freqs: np.ndarray, amps: np.ndarray, top_k: int = 3) -> Dict[str, object]:
    top_freqs, top_amps = detect_main_peaks(freqs, amps, top_k=top_k)
    main_freq = float(top_freqs[0]) if len(top_freqs) else 0.0
    return {
        "freqs": freqs,
        "amps": amps,
        "top_freqs": top_freqs,
        "top_amps": top_amps,
        "main_freq": main_freq,
        "harmonics": harmonic_candidates(main_freq),
        "peak_confidence": peak_confidence(amps, top_amps),
    }


def analyze_frequency(signal: np.ndarray, fs: int, top_k: int = 3) -> Dict[str, object]:
    freqs, amps = compute_fft(signal, fs)
    return summarize_spectrum(freqs, amps, top_k=top_k)
