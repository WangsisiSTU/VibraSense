from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.orchestration import run_analysis
from src.predict import get_modality_summary
from src.report import serialize_report_json


def _plot_signal(title: str, signal: np.ndarray):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(signal)
    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)


def _plot_spectrum(freqs: np.ndarray, amps: np.ndarray, x_max: float):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(freqs, amps)
    ax.set_xlim(0, x_max)
    ax.set_title("FFT Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)


def _render_modality_channel(title: str, raw_signal: np.ndarray, filtered_signal: np.ndarray, frequency: dict, x_max: float):
    st.markdown(f"### {title}")
    _plot_signal(f"Raw {title} Signal", raw_signal)
    _plot_signal(f"Filtered {title} Signal", filtered_signal)
    _plot_spectrum(
        frequency["freqs"],
        frequency["amps"],
        x_max=x_max,
    )
    top = frequency["top_freqs"]
    st.write(f"{title} 主峰频率:", ", ".join([f"{v:.1f} Hz" for v in top[:3]]) if len(top) else "N/A")


def _render_anomaly_cards(modality_results: dict):
    st.subheader("2) 异常检测")
    anomaly_cards = st.columns(3)
    for idx, modality in enumerate(["audio", "vibration", "sensor"]):
        with anomaly_cards[idx]:
            if modality in modality_results:
                data = get_modality_summary(modality_results[modality])
                st.markdown(f"**{modality.title()}**")
                st.write(f"Label: `{data['label']}`")
                st.write(f"Score: `{data['score']:.3f}`")
                st.write("Evidence:")
                for item in data.get("evidence", [])[:3]:
                    st.write(f"- {item}")
            else:
                st.markdown(f"**{modality.title()}**")
                st.write("未提供该模态数据")


def main():
    st.set_page_config(page_title="VibraSense AI", layout="wide")
    st.title("VibraSense AI")
    st.subheader("通用时序信号智能引擎（MVP）")
    st.markdown("支持输入：`wav` 音频、`csv` 振动、`csv` 传感器流。")

    col_left, col_right = st.columns(2)
    with col_left:
        audio_file = st.file_uploader("上传音频文件 (.wav)", type=["wav"])
        vibration_file = st.file_uploader("上传振动文件 (.csv)", type=["csv"])
    with col_right:
        vibration_fs = st.number_input("振动采样率 (Hz)", min_value=100, max_value=96000, value=12000, step=100)
        sensor_file = st.file_uploader("上传传感器流 (.csv)", type=["csv"])

    run_button = st.button("运行 VibraSense 分析")
    if not run_button:
        return

    analysis = run_analysis(
        audio_file=audio_file,
        vibration_file=vibration_file,
        vibration_fs=int(vibration_fs),
        sensor_file=sensor_file,
    )
    modality_results = analysis["modality_results"]
    raw_outputs = analysis["raw_outputs"]
    fusion_result = analysis["fusion_result"]
    report_text = analysis["report_text"]
    report_json = analysis["report_json"]

    if not modality_results:
        st.warning("请至少上传一种输入数据（音频/振动/传感器）。")
        st.stop()

    st.subheader("1) 降噪结果与频谱")
    if "audio" in modality_results:
        _render_modality_channel(
            "Audio",
            raw_outputs["audio"]["signal"],
            modality_results["audio"]["filtered_signal"],
            modality_results["audio"]["frequency"],
            x_max=5000,
        )

    if "vibration" in modality_results:
        _render_modality_channel(
            "Vibration",
            raw_outputs["vibration"]["signal"],
            modality_results["vibration"]["filtered_signal"],
            modality_results["vibration"]["frequency"],
            x_max=min(5000, int(vibration_fs) // 2),
        )

    _render_anomaly_cards(modality_results)

    st.subheader("3) 风险评分")
    st.metric("Risk Score", f"{fusion_result['risk_score']:.2f} / 100")
    st.write(f"Risk Level: **{fusion_result['risk_level']}**")
    st.json(fusion_result["contributors"])

    st.subheader("4) 工程解释报告")
    st.markdown(report_text)
    st.download_button(
        label="下载报告 Markdown",
        data=report_text,
        file_name="vibrasense_report.md",
        mime="text/markdown",
    )
    st.download_button(
        label="下载报告 JSON",
        data=serialize_report_json(report_json),
        file_name="vibrasense_report.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
