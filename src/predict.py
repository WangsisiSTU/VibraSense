from __future__ import annotations

from typing import Dict

from src.fft_tools import analyze_frequency
from src.features import extract_features
from src.fusion import fuse_scores as _fuse_scores
from src.model_audio import predict_anomaly as predict_audio_anomaly
from src.model_vibration import predict_anomaly as predict_vibration_anomaly
from src.preprocess import preprocess_signal
from src.sensor import evaluate_sensor_stream as _evaluate_sensor_stream


SUMMARY_KEYS = (
    "score",
    "label",
    "evidence",
    "confidence",
    "rule_score",
    "unsup_score",
    "inference_mode",
)


def get_modality_summary(result: Dict[str, object]) -> Dict[str, object]:
    if not result:
        return {}
    if "anomaly" in result and isinstance(result["anomaly"], dict):
        return result["anomaly"]
    return {key: result[key] for key in SUMMARY_KEYS if key in result}


def build_fusion_input(modality_results: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {
        modality: get_modality_summary(result)
        for modality, result in modality_results.items()
        if result
    }


def evaluate_sensor_stream(sensor_df) -> Dict[str, object]:
    return _evaluate_sensor_stream(sensor_df)


def fuse_scores(modality_outputs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    return _fuse_scores(modality_outputs)


def run_modality_pipeline(signal, fs: int, modality: str, model_artifact: Dict[str, object] | None = None) -> Dict[str, object]:
    filtered_signal, preprocess_meta = preprocess_signal(signal, fs, modality)
    freq_result = analyze_frequency(filtered_signal, fs)
    features = extract_features(filtered_signal, fs, modality, spectrum=freq_result)

    if modality == "audio":
        anomaly = predict_audio_anomaly(filtered_signal, fs, features, model_artifact=model_artifact)
    else:
        anomaly = predict_vibration_anomaly(filtered_signal, fs, features, model_artifact=model_artifact)

    result = {
        "modality": modality,
        "filtered_signal": filtered_signal,
        "preprocess_meta": preprocess_meta,
        "frequency": freq_result,
        "features": features,
        "anomaly": anomaly,
    }
    result.update({key: value for key, value in anomaly.items() if key in SUMMARY_KEYS})
    return result
