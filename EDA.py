import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, fftfreq

# 低通滤波器函数
def lowpass_filter(data, cutoff=5, fs=500, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# 基于 SCL 提取基线
def extract_scl_baseline(signal, sampling_rate):
    baseline_mean = np.mean(signal)
    baseline_std = np.std(signal)
    return baseline_mean, baseline_std

# SCR 波动频率计算
def calculate_scr_features(signal, sampling_rate):
    # 自适应设定最小幅值阈值为信号标准差的倍数
    min_amplitude = np.std(signal) * 1
    peaks, properties = find_peaks(signal, height=min_amplitude)
    peak_intervals = np.diff(peaks) / sampling_rate  # 峰值间隔
    scr_frequency = len(peaks) / (len(signal) / sampling_rate / 3 )  # 每分钟波动次数
    return scr_frequency, peaks, properties["peak_heights"]

# 压力判定函数
def stress_analysis(signal, baseline_mean, baseline_std, sampling_rate, window_size=20):
    # 滑动窗口大小，覆盖率 50%
    window_samples = window_size * sampling_rate
    step_size = window_samples // 2  # 50% 覆盖率
    num_windows = (len(signal) - window_samples) // step_size + 1

    stress_results = []
    time_stamps = []
    scl_means = []
    scr_frequencies = []
    all_peak_heights = []

    for i in range(num_windows):
        start = i * step_size
        end = start + window_samples
        window_data = signal[start:end]

        # 时间戳
        time_stamps.append(start / sampling_rate)

        # SCL 均值计算
        scl_mean = np.mean(window_data)
        scl_stress = scl_mean > (baseline_mean + 2 * baseline_std)

        # SCR 特征提取
        scr_frequency, _, peak_heights = calculate_scr_features(window_data, sampling_rate)
        scr_stress = scr_frequency > 5  # 自定义阈值

        # 综合压力判断
        stress_detected = scl_stress and scr_stress
        stress_results.append({
            "window": i + 1,
            "scl_mean": scl_mean,
            "scl_stress": scl_stress,
            "scr_frequency": scr_frequency,
            "scr_stress": scr_stress,
            "peak_heights": peak_heights.tolist(),
            "stress_detected": stress_detected,
        })

        # 保存中间特征值
        scl_means.append(scl_mean)
        scr_frequencies.append(scr_frequency)
        all_peak_heights.append(np.mean(peak_heights) if len(peak_heights) > 0 else 0)

    return stress_results, time_stamps, scl_means, scr_frequencies, all_peak_heights

# 从静息阶段文件提取基线
def load_baseline_from_file(file_path, channel, sampling_rate):
    baseline_df = pd.read_csv(file_path, sep='\s+', comment='#', header=None)
    baseline_df.columns = ['nSeq', 'DI', 'CH1', 'CH2', 'CH3']  # 根据文件实际列名调整
    baseline_signal = baseline_df[channel]
    filtered_baseline_signal = lowpass_filter(baseline_signal, cutoff=5, fs=sampling_rate)

    # 绘制静息阶段的 SCL 曲线
    time = np.arange(len(filtered_baseline_signal)) / sampling_rate
    return filtered_baseline_signal, time

# 文件路径
file_path_1 = r'/Users/liu/Downloads/new/10/opensignals_00078065DFF4_2024-08-28_14-31-04.txt'
file_path_2 = r'/Users/liu/Downloads/new/10/opensignals_00078065DFF4_2024-08-28_14-35-00.txt'
file_path_3 = r'/Users/liu/Downloads/new/10/opensignals_00078065DFF4_2024-08-28_14-51-20.txt'
file_path_4 = r'/Users/liu/Downloads/new/10/opensignals_00078065DFF4_2024-08-28_15-06-58.txt'
baseline_file_path = r'/Users/liu/Downloads/new/10/opensignals_00078065DFF4_2024-08-28_14-31-04.txt'  # 静息阶段数据文件路径

# 跳过头部读取数据
df_1 = pd.read_csv(file_path_1, sep='\s+', comment='#', header=None)
df_2 = pd.read_csv(file_path_2, sep='\s+', comment='#', header=None)
df_3 = pd.read_csv(file_path_3, sep='\s+', comment='#', header=None)
df_4 = pd.read_csv(file_path_4, sep='\s+', comment='#', header=None)
df_1.columns = ['nSeq', 'DI', 'CH1', 'CH2', 'CH3']
df_2.columns = ['nSeq', 'DI', 'CH1', 'CH2', 'CH3']
df_3.columns = ['nSeq', 'DI', 'CH1', 'CH2', 'CH3']
df_4.columns = ['nSeq', 'DI', 'CH1', 'CH2', 'CH3']

# 选择 CH3 通道的 EDA 信号
signal_1 = df_1['CH3']
signal_2 = df_2['CH3']
signal_3 = df_3['CH3']
signal_4 = df_4['CH3']

# 数据预处理
sampling_rate = 500  # 采样率（Hz）
filtered_signal_1 = lowpass_filter(signal_1, cutoff=5, fs=sampling_rate)
filtered_signal_2 = lowpass_filter(signal_2, cutoff=5, fs=sampling_rate)
filtered_signal_3 = lowpass_filter(signal_3, cutoff=5, fs=sampling_rate)
filtered_signal_4 = lowpass_filter(signal_4, cutoff=5, fs=sampling_rate)

# 从静息阶段文件计算基线
filtered_baseline_signal, baseline_time = load_baseline_from_file(baseline_file_path, 'CH3', sampling_rate)
baseline_mean = np.mean(filtered_baseline_signal)
baseline_std = np.std(filtered_baseline_signal)

# 压力分析
results_1, time_stamps_1, scl_means_1, scr_frequencies_1, all_peak_heights_1 = stress_analysis(filtered_signal_1, baseline_mean, baseline_std, sampling_rate)
results_2, time_stamps_2, scl_means_2, scr_frequencies_2, all_peak_heights_2 = stress_analysis(filtered_signal_2, baseline_mean, baseline_std, sampling_rate)
results_3, time_stamps_3, scl_means_3, scr_frequencies_3, all_peak_heights_3 = stress_analysis(filtered_signal_3, baseline_mean, baseline_std, sampling_rate)
results_4, time_stamps_4, scl_means_4, scr_frequencies_4, all_peak_heights_4 = stress_analysis(filtered_signal_4, baseline_mean, baseline_std, sampling_rate)

# 绘制 SCL 和 SCR 频率随时间的趋势图（双Y轴）
plt.figure(figsize=(15, 12))

# 数据 1
ax1 = plt.subplot(2, 2, 1)
ax2 = ax1.twinx()
line1, = ax1.plot(time_stamps_1, scl_means_1, label="SCL Mean", color="blue")
line2, = ax2.plot(time_stamps_1, scr_frequencies_1, label="SCR Frequency", color="orange")
for i, res in enumerate(results_1):
    if res["stress_detected"]:
        ax1.axvspan(time_stamps_1[i], time_stamps_1[i] + 10, color='red', alpha=0.5)
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("SCL Mean", color="blue")
ax2.set_ylabel("SCR Frequency", color="orange")
ax1.set_title("a) Still zone", loc='left')
ax1.grid()
ax1.legend([line1, line2], ["SCL Mean", "SCR Frequency"], loc='upper right')

# 数据 2
ax1 = plt.subplot(2, 2, 2)
ax2 = ax1.twinx()
line1, = ax1.plot(time_stamps_2, scl_means_2, label="SCL Mean", color="blue")
line2, = ax2.plot(time_stamps_2, scr_frequencies_2, label="SCR Frequency", color="orange")
for i, res in enumerate(results_2):
    if res["stress_detected"]:
        ax1.axvspan(time_stamps_2[i], time_stamps_2[i] + 10, color='red', alpha=0.5)
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("SCL Mean", color="blue")
ax2.set_ylabel("SCR Frequency", color="orange")
ax1.set_title("b) 30-zone", loc='left')
ax1.grid()
ax1.legend([line1, line2], ["SCL Mean", "SCR Frequency"], loc='upper right')

# 数据 3
ax1 = plt.subplot(2, 2, 3)
ax2 = ax1.twinx()
line1, = ax1.plot(time_stamps_3, scl_means_3, label="SCL Mean", color="blue")
line2, = ax2.plot(time_stamps_3, scr_frequencies_3, label="SCR Frequency", color="orange")
for i, res in enumerate(results_3):
    if res["stress_detected"]:
        ax1.axvspan(time_stamps_3[i], time_stamps_3[i] + 10, color='red', alpha=0.5)
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("SCL Mean", color="blue")
ax2.set_ylabel("SCR Frequency", color="orange")
ax1.set_title("c) 50-zone", loc='left')
ax1.grid()
ax1.legend([line1, line2], ["SCL Mean", "SCR Frequency"], loc='upper right')

# 数据 4
ax1 = plt.subplot(2, 2, 4)
ax2 = ax1.twinx()
line1, = ax1.plot(time_stamps_4, scl_means_4, label="SCL Mean", color="blue")
line2, = ax2.plot(time_stamps_4, scr_frequencies_4, label="SCR Frequency", color="orange")
for i, res in enumerate(results_4):
    if res["stress_detected"]:
        ax1.axvspan(time_stamps_4[i], time_stamps_4[i] + 10, color='red', alpha=0.5)
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("SCL Mean", color="blue")
ax2.set_ylabel("SCR Frequency", color="orange")
ax1.set_title("d) Highway", loc='left')
ax1.grid()
ax1.legend([line1, line2], ["SCL Mean", "SCR Frequency"], loc='upper right')

plt.tight_layout()
plt.show()

# 输出每段总压力时间
stress_total_duration_1 = sum([1 if res["stress_detected"] else 0 for res in results_1]) * 10
stress_total_duration_2 = sum([1 if res["stress_detected"] else 0 for res in results_2]) * 10
stress_total_duration_3 = sum([1 if res["stress_detected"] else 0 for res in results_3]) * 10
stress_total_duration_4 = sum([1 if res["stress_detected"] else 0 for res in results_4]) * 10

print(f"Total Stress Duration (Data 1): {stress_total_duration_1} seconds")
print(f"Total Stress Duration (Data 2): {stress_total_duration_2} seconds")
print(f"Total Stress Duration (Data 3): {stress_total_duration_3} seconds")
print(f"Total Stress Duration (Data 4): {stress_total_duration_4} seconds")
