# generate_test_samples.py
# 运行后自动生成：
# 1. normal_motor.wav
# 2. abnormal_motor.wav
# 3. building_vibration.csv

import numpy as np
import pandas as pd
from scipy.io.wavfile import write

# ===============================
# 参数设置
# ===============================
fs_audio = 16000          # 音频采样率
duration = 3             # 秒
t = np.linspace(0, duration, fs_audio * duration, endpoint=False)

# ===============================
# 样本1：正常设备音频
# 主频60Hz + 少量噪声
# ===============================
normal = (
    0.8 * np.sin(2 * np.pi * 60 * t) +
    0.03 * np.random.randn(len(t))
)

normal = normal / np.max(np.abs(normal))
write("normal_motor.wav", fs_audio, (normal * 32767).astype(np.int16))

# ===============================
# 样本2：异常设备音频
# 60Hz主频 + 1800Hz高频异常 + 冲击
# ===============================
abnormal = (
    0.8 * np.sin(2 * np.pi * 60 * t) +
    0.25 * np.sin(2 * np.pi * 1800 * t) +
    0.04 * np.random.randn(len(t))
)

# 加入周期冲击
for i in range(5):
    idx = np.random.randint(0, len(t)-300)
    abnormal[idx:idx+300] += np.hanning(300) * 0.8

abnormal = abnormal / np.max(np.abs(abnormal))
write("abnormal_motor.wav", fs_audio, (abnormal * 32767).astype(np.int16))

# ===============================
# 样本3：楼宇振动 CSV
# 一阶频率1.8Hz + 随机噪声
# ===============================
fs_vib = 100
tv = np.linspace(0, 60, fs_vib * 60, endpoint=False)

vibration = (
    0.02 * np.sin(2 * np.pi * 1.8 * tv) +
    0.003 * np.random.randn(len(tv))
)

df = pd.DataFrame({
    "time_sec": tv,
    "acceleration": vibration
})

df.to_csv("building_vibration.csv", index=False)

print("已生成测试文件：")
print("1. normal_motor.wav")
print("2. abnormal_motor.wav")
print("3. building_vibration.csv")