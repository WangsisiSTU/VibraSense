# VibraSense-AI

通用时序信号智能引擎（MVP），支持多模态输入：

- 音频 `wav`
- 振动 `csv`
- 传感器流 `csv`

输出结果：

- 降噪结果
- 主频识别
- 异常检测
- 风险评分
- 工程解释报告（Markdown + JSON）

## Project Structure

```text
VibraSense-AI/
│── app.py
│── main.py
│── requirements.txt
│── README.md
│
├── data/
│   ├── audio/
│   └── vibration/
│
├── src/
│   ├── preprocess.py
│   ├── fft_tools.py
│   ├── features.py
│   ├── model_audio.py
│   ├── model_vibration.py
│   ├── predict.py
│   └── report.py
│
└── models/
```

## Quick Start

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动应用

```bash
streamlit run app.py
```

兼容入口：

```bash
streamlit run main.py
```

## Input Data Notes

- `wav`：单通道音频会自动读取；多通道默认转 mono。
- 振动 `csv`：默认读取第一个数值列作为信号，采样率在页面中填写。
- 传感器流 `csv`：会对所有数值列进行异常漂移与离群检测。

## Scoring Strategy (MVP)

- 模态内：规则分 + 无监督异常分（IsolationForest）融合。
- 模态间：按置信度修正权重后做风险融合。
- 风险等级：`Low / Medium / High`，并输出贡献项解释。

## Dataset Background

- 音频主数据：AIST 工业设备异常声音（风扇、泵、阀门）
- 振动辅数据：Case Western Reserve University 轴承故障数据集

## Next Step Suggestions

- 在 `models/` 持久化训练得到的模型参数（joblib）。
- 引入更稳定的特征标准化和基线数据管理。
- 扩展在线流处理（socket/MQ）和实时告警。
