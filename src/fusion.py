from __future__ import annotations

from typing import Dict

import numpy as np


def fuse_scores(modality_outputs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    base_weights = {"audio": 0.4, "vibration": 0.4, "sensor": 0.2}
    weighted_scores = []
    contributors = {}
    total_weight = 0.0

    for modality, result in modality_outputs.items():
        if not result:
            continue
        weight = base_weights.get(modality, 0.2)
        confidence = float(result.get("confidence", 0.7))
        effective_weight = weight * np.clip(confidence, 0.3, 1.0)
        score = float(result.get("score", 0.0))
        weighted_scores.append(score * effective_weight)
        total_weight += effective_weight
        contributors[modality] = {
            "raw_score": score,
            "weight": effective_weight,
            "label": result.get("label", "unknown"),
        }

    if total_weight <= 1e-8:
        risk_score = 0.0
    else:
        risk_score = float(np.clip(sum(weighted_scores) / total_weight, 0.0, 1.0))

    scaled = round(risk_score * 100.0, 2)
    level = "Low" if scaled < 35 else "Medium" if scaled < 65 else "High"
    return {"risk_score": scaled, "risk_level": level, "contributors": contributors}
