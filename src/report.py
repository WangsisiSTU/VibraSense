from __future__ import annotations

import json
from typing import Dict, List, Tuple


def map_frequency_interpretation(main_freq: float) -> str:
    if main_freq <= 0:
        return "主频信息不足，建议延长采集时长并复测。"
    if main_freq < 80:
        return "低频主峰明显，可能存在松动、基础激励或低阶共振。"
    if main_freq < 500:
        return "中频主峰突出，常见于旋转部件激励或结构耦合。"
    return "高频主峰占优，可能涉及摩擦、磨损或局部冲击。"


def _collect_evidence(modality_results: Dict[str, Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for modality, result in modality_results.items():
        anomaly = _report_summary(result)
        evidence = anomaly.get("evidence", [])
        for item in evidence[:3]:
            lines.append(f"[{modality}] {item}")
    return lines


def _report_summary(result: Dict[str, object]) -> Dict[str, object]:
    if "anomaly" in result and isinstance(result["anomaly"], dict):
        return result["anomaly"]
    return result


def generate_report(
    modality_results: Dict[str, Dict[str, object]],
    fusion_result: Dict[str, object],
) -> Tuple[str, Dict[str, object]]:
    evidence_lines = _collect_evidence(modality_results)
    main_freq_audio = (
        float(modality_results["audio"]["frequency"]["main_freq"])
        if "audio" in modality_results and modality_results["audio"]
        else 0.0
    )
    main_freq_vib = (
        float(modality_results["vibration"]["frequency"]["main_freq"])
        if "vibration" in modality_results and modality_results["vibration"]
        else 0.0
    )

    dominant = max(main_freq_audio, main_freq_vib)
    freq_explain = map_frequency_interpretation(dominant)

    risk_score = fusion_result.get("risk_score", 0.0)
    risk_level = fusion_result.get("risk_level", "Low")
    contributor_lines = []
    for m, info in fusion_result.get("contributors", {}).items():
        contributor_lines.append(
            f"- {m}: score={info['raw_score']:.2f}, weight={info['weight']:.2f}, label={info['label']}"
        )

    report_lines = [
        "# VibraSense Engineering Report",
        "",
        "## 1) Risk Overview",
        f"- Overall risk score: **{risk_score} / 100**",
        f"- Risk level: **{risk_level}**",
        "",
        "## 2) Main Frequency Diagnosis",
        f"- Audio main frequency: {main_freq_audio:.2f} Hz",
        f"- Vibration main frequency: {main_freq_vib:.2f} Hz",
        f"- Engineering interpretation: {freq_explain}",
        "",
        "## 3) Anomaly Evidence",
    ]
    if evidence_lines:
        report_lines.extend([f"- {line}" for line in evidence_lines])
    else:
        report_lines.append("- No strong anomaly evidence.")

    report_lines.extend(["", "## 4) Risk Contributors"])
    if contributor_lines:
        report_lines.extend(contributor_lines)
    else:
        report_lines.append("- No modality contributors available.")

    report_lines.extend(
        [
            "",
            "## 5) Suggested Actions",
            "- Medium/High risk: inspect rotating parts, bearings and mounting bolts first.",
            "- If high-frequency peaks persist, schedule lubrication and wear inspection.",
            "- Continue collecting baseline normal data to improve unsupervised model stability.",
        ]
    )
    markdown_text = "\n".join(report_lines)

    json_summary = {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "audio_main_freq": main_freq_audio,
        "vibration_main_freq": main_freq_vib,
        "evidence": evidence_lines,
        "contributors": fusion_result.get("contributors", {}),
    }
    return markdown_text, json_summary


def serialize_report_json(report_json: Dict[str, object]) -> str:
    return json.dumps(report_json, ensure_ascii=False, indent=2)
