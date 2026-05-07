from __future__ import annotations

from typing import Dict, List

from src.anomaly_utils import (
    build_window_feature_matrix,
    combine_scores,
    label_from_score,
    score_window_matrix,
)
from src.features import extract_features


VIBRATION_WINDOW_FEATURES = (
    "rms",
    "crest_factor",
    "spectral_centroid",
    "spectral_bandwidth",
    "kurtosis",
    "energy_high_ratio",
)
VIBRATION_WINDOWS = 16


def rule_score(features: Dict[str, float]) -> Dict[str, object]:
    score = 0.0
    evidence: List[str] = []
    if features["crest_factor"] > 5.0:
        score += 0.35
        evidence.append("Crest factor suggests impulsive vibration.")
    if features["kurtosis"] > 4.5:
        score += 0.25
        evidence.append("Kurtosis indicates potential bearing impacts.")
    if features["energy_high_ratio"] > 0.35:
        score += 0.20
        evidence.append("High-band energy ratio is elevated.")
    if features["rms"] > 0.2:
        score += 0.15
        evidence.append("RMS exceeds vibration baseline threshold.")
    return {"rule_score": float(min(score, 1.0)), "evidence": evidence}


def model_score(signal, fs: int, model_artifact: Dict[str, object] | None = None) -> Dict[str, object]:
    matrix = build_window_feature_matrix(
        signal,
        fs,
        "vibration",
        windows=VIBRATION_WINDOWS,
        feature_keys=VIBRATION_WINDOW_FEATURES,
    )
    result = score_window_matrix(
        matrix,
        n_estimators=150,
        contamination=0.2,
        min_rows=10,
        fallback_score=0.3,
        fallback_confidence=0.65,
        model_artifact=model_artifact,
    )
    return {"window_matrix": matrix, **result}


def predict_anomaly(signal, fs: int, features: Dict[str, float] | None = None, model_artifact: Dict[str, object] | None = None) -> Dict[str, object]:
    if features is None:
        features = extract_features(signal, fs, modality="vibration")

    rule_result = rule_score(features)
    model_result = model_score(signal, fs, model_artifact=model_artifact)
    evidence = list(rule_result["evidence"])
    if model_result["mode"] == "fallback":
        evidence.append("Limited windows, fallback unsupervised score used.")
    elif model_result["mode"] == "batch_fit":
        evidence.append("Unsupervised score fitted on the current sample batch.")

    score = combine_scores(
        float(rule_result["rule_score"]),
        float(model_result["unsup_score"]),
        rule_weight=0.6,
    )
    label = label_from_score(score)

    if not evidence:
        evidence.append("No strong anomaly evidence from vibration features.")

    return {
        "modality": "vibration",
        "score": score,
        "label": label,
        "evidence": evidence,
        "rule_score": float(rule_result["rule_score"]),
        "unsup_score": float(model_result["unsup_score"]),
        "confidence": float(model_result["confidence"]),
        "inference_mode": model_result["mode"],
    }
