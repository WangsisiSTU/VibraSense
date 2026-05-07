from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.ensemble import IsolationForest

from src.features import extract_features
from src.fft_tools import analyze_frequency


def build_window_feature_matrix(
    signal: np.ndarray,
    fs: int,
    modality: str,
    *,
    windows: int,
    feature_keys: Iterable[str],
) -> np.ndarray:
    chunks = np.array_split(signal, windows)
    rows: List[List[float]] = []
    keys = list(feature_keys)
    for chunk in chunks:
        if len(chunk) < 8:
            continue
        spectrum = analyze_frequency(chunk, fs)
        feats = extract_features(chunk, fs, modality, spectrum=spectrum)
        rows.append([feats[key] for key in keys])
    return np.asarray(rows, dtype=float)


def label_from_score(score: float, *, warning_threshold: float = 0.35, anomalous_threshold: float = 0.65) -> str:
    if score < warning_threshold:
        return "normal"
    if score < anomalous_threshold:
        return "warning"
    return "anomalous"


def combine_scores(rule_score: float, unsup_score: float, *, rule_weight: float) -> float:
    unsup_weight = 1.0 - rule_weight
    return float(np.clip(rule_weight * rule_score + unsup_weight * unsup_score, 0.0, 1.0))


def score_window_matrix(
    window_matrix: np.ndarray,
    *,
    n_estimators: int,
    contamination: float,
    min_rows: int,
    fallback_score: float,
    fallback_confidence: float,
    model_artifact: Dict[str, object] | None = None,
    random_state: int = 42,
) -> Dict[str, object]:
    if len(window_matrix) < min_rows:
        return {
            "unsup_score": fallback_score,
            "confidence": fallback_confidence,
            "mode": "fallback",
        }

    transformed = window_matrix
    mode = "batch_fit"
    if model_artifact:
        scaler = model_artifact.get("scaler")
        model = model_artifact.get("model")
        if scaler is not None:
            transformed = scaler.transform(window_matrix)
        if model is not None:
            anomaly_raw = -model.score_samples(transformed)
            mode = "loaded_model"
        else:
            model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=random_state,
            )
            model.fit(transformed)
            anomaly_raw = -model.score_samples(transformed)
    else:
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        model.fit(transformed)
        anomaly_raw = -model.score_samples(transformed)

    unsup_score = float(
        np.clip(
            (np.mean(anomaly_raw) - np.min(anomaly_raw)) / (np.ptp(anomaly_raw) + 1e-8),
            0.0,
            1.0,
        )
    )
    confidence = 0.9 if mode == "loaded_model" else 0.75
    return {
        "unsup_score": unsup_score,
        "confidence": confidence,
        "mode": mode,
    }
