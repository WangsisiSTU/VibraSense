from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.anomaly_utils import label_from_score


def evaluate_sensor_stream(sensor_df: pd.DataFrame) -> Dict[str, object]:
    numeric_cols = sensor_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {
            "modality": "sensor",
            "score": 0.0,
            "label": "normal",
            "evidence": ["No numeric sensor columns found."],
            "confidence": 0.3,
            "contributors": {},
        }

    score = 0.0
    evidence: List[str] = []
    per_col = {}
    for col in numeric_cols:
        values = sensor_df[col].astype(float).to_numpy()
        if len(values) < 5:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values)) + 1e-8
        zmax = float(np.max(np.abs(values - mean) / std))
        drift = float(np.mean(np.diff(values)))
        col_score = min(1.0, 0.08 * zmax + min(abs(drift) * 2.0, 0.3))
        per_col[col] = col_score
        score += col_score / max(1, len(numeric_cols))
        if zmax > 4.0:
            evidence.append(f"{col}: large outlier deviation detected (zmax={zmax:.2f}).")
        if abs(drift) > 0.02:
            evidence.append(f"{col}: trend drift observed (drift={drift:.4f}).")

    score = float(np.clip(score, 0.0, 1.0))
    if not evidence:
        evidence.append("Sensor stream remains within expected trend and variance.")

    return {
        "modality": "sensor",
        "score": score,
        "label": label_from_score(score, warning_threshold=0.3, anomalous_threshold=0.6),
        "evidence": evidence,
        "confidence": 0.8,
        "contributors": per_col,
    }
