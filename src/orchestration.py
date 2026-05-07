from __future__ import annotations

from typing import Any, Dict

from src.predict import build_fusion_input, evaluate_sensor_stream, fuse_scores, run_modality_pipeline
from src.preprocess import load_audio, load_csv_signal, load_sensor_stream_csv
from src.report import generate_report


def run_analysis(
    *,
    audio_file: Any = None,
    vibration_file: Any = None,
    vibration_fs: int = 12000,
    sensor_file: Any = None,
) -> Dict[str, object]:
    modality_results: Dict[str, Dict[str, object]] = {}
    raw_outputs: Dict[str, Dict[str, object]] = {}

    if audio_file is not None:
        audio_raw, audio_fs, audio_meta = load_audio(audio_file)
        raw_outputs["audio"] = {"signal": audio_raw, "fs": audio_fs, "meta": audio_meta}
        modality_results["audio"] = run_modality_pipeline(audio_raw, audio_fs, modality="audio")

    if vibration_file is not None:
        vib_raw, vib_meta = load_csv_signal(vibration_file)
        vibration_fs = int(vibration_fs)
        raw_outputs["vibration"] = {"signal": vib_raw, "fs": vibration_fs, "meta": vib_meta}
        modality_results["vibration"] = run_modality_pipeline(vib_raw, vibration_fs, modality="vibration")

    if sensor_file is not None:
        sensor_df, sensor_meta = load_sensor_stream_csv(sensor_file)
        sensor_result = evaluate_sensor_stream(sensor_df)
        sensor_result["meta"] = sensor_meta
        modality_results["sensor"] = sensor_result

    fusion_result = fuse_scores(build_fusion_input(modality_results))
    report_text, report_json = generate_report(modality_results, fusion_result)

    return {
        "modality_results": modality_results,
        "raw_outputs": raw_outputs,
        "fusion_result": fusion_result,
        "report_text": report_text,
        "report_json": report_json,
    }
