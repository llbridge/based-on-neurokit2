import numpy as np
import pandas as pd
import pywt
import neurokit2 as nk
from scipy.stats import norm
from scipy.signal import find_peaks, butter, filtfilt
from neurokit2.hrv.hrv_utils import _hrv_format_input
import matplotlib.pyplot as plt

# 文件路径
file_paths = {
    "a) Still Zone": r'/Users/liu/Downloads/new/03/opensignals_00078065DFF4_2024-08-15_11-47-34.txt',
    "b) 30-zone": r'/Users/liu/Downloads/new/03/opensignals_00078065DFF4_2024-08-15_11-51-22.txt',
    "c) 50-zone": r'/Users/liu/Downloads/new/03/opensignals_00078065DFF4_2024-08-15_12-07-32.txt',
    "d) Highway": r'/Users/liu/Downloads/new/03/opensignals_00078065DFF4_2024-08-15_12-18-04.txt'
}

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
    min_amplitude = np.std(signal) * 1
    peaks, properties = find_peaks(signal, height=min_amplitude)
    scr_frequency = len(peaks) / (len(signal) / sampling_rate / 3)
    return scr_frequency, peaks, properties["peak_heights"]

# EEG 小波分解提取 Alpha 和 Beta 能量
def wavelet_decomposition_alpha_beta(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    alpha_coeffs = coeffs[level - 2] 
    beta_coeffs = coeffs[level - 3]  
    alpha_energy = np.sum(np.square(alpha_coeffs))
    beta_energy = np.sum(np.square(beta_coeffs))
    return alpha_energy, beta_energy

# 滑动窗口处理 EEG 信号
def process_with_sliding_window(signal, window_size, step_size, sampling_rate, wavelet='db4', level=4):
    window_samples = int(window_size * sampling_rate)
    step_samples = int(step_size * sampling_rate)
    num_windows = (len(signal) - window_samples) // step_samples + 1

    alpha_energies = []
    beta_energies = []

    for i in range(num_windows):
        start = i * step_samples
        end = start + window_samples
        segment = signal[start:end]
        alpha_energy, beta_energy = wavelet_decomposition_alpha_beta(segment, wavelet=wavelet, level=level)
        alpha_energies.append(alpha_energy)
        beta_energies.append(beta_energy)

    return alpha_energies, beta_energies

# 动态压力检测
def calculate_dynamic_stress(alpha_energies, beta_energies, window_size=20, k=2.25):
    ratios = [b / a for a, b in zip(alpha_energies, beta_energies)]
    stress_indices = []
    thresholds = []

    for i in range(len(ratios)):
        start = max(0, i - window_size + 1)
        window = ratios[start:i + 1]
        mu = np.mean(window)
        sigma = np.std(window)
        threshold = mu + k * sigma
        thresholds.append(threshold)
        stress_indices.append(ratios[i] > threshold)

    return ratios, thresholds, stress_indices

# 压力判定函数
def stress_analysis(signal, baseline_mean, baseline_std, sampling_rate, window_size=20):
    window_samples = window_size * sampling_rate
    step_size = window_samples // 2  # 50% 覆盖率
    num_windows = (len(signal) - window_samples) // step_size + 1

    stress_results = []
    scl_means = []
    scr_frequencies = []
    all_peak_heights = []

    for i in range(num_windows):
        start = i * step_size
        end = start + window_samples
        window_data = signal[start:end]

        scl_mean = np.mean(window_data)
        scl_stress = scl_mean > (baseline_mean + 2 * baseline_std)

        scr_frequency, _, peak_heights = calculate_scr_features(window_data, sampling_rate)
        scr_stress = scr_frequency > 4.0  # 自定义阈值

        stress_detected = scl_stress and scr_stress
        stress_results.append({
            "scl_mean": scl_mean,
            "scl_stress": scl_stress,
            "scr_frequency": scr_frequency,
            "scr_stress": scr_stress,
            "stress_detected": stress_detected,
        })

        scl_means.append(scl_mean)
        scr_frequencies.append(scr_frequency)
        all_peak_heights.append(np.mean(peak_heights) if len(peak_heights) > 0 else 0)

    return stress_results, scl_means, scr_frequencies, all_peak_heights

# 初始化变量
all_time_axes = []
final_stress_flags = []
stress_durations = []
stress_percentages = []

# 提取基线
baseline_file_path = file_paths["a) Still Zone"]
baseline_df = pd.read_csv(baseline_file_path, sep='\s+', comment='#', header=None)
baseline_df.columns = ['nSeq', 'DI', 'CH1', 'CH2', 'CH3']
baseline_signal = baseline_df['CH3']
filtered_baseline_signal = lowpass_filter(baseline_signal, cutoff=5, fs=500)
baseline_mean, baseline_std = extract_scl_baseline(filtered_baseline_signal, sampling_rate=500)

# 处理每个区域数据
for zone, file_path in file_paths.items():
    df = pd.read_csv(file_path, sep='\s+', comment='#', header=None)
    df.columns = ['nSeq', 'DI', 'CH1', 'CH2', 'CH3']

    # ECG 处理
    ecg_raw = df['CH1']
    ecg = ecg_raw.tolist()
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=500)
    peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=500, correct_artifacts=True)

    rri, rri_time, rri_missing = _hrv_format_input(peaks, sampling_rate=500)
    hr = 60000 / np.array(rri)  # 计算心率 (HR)

    mu_all, sigma_all = norm.fit(rri)
    peaks = peaks.to_numpy()
    window_len = 20
    res = []
    time_axis = []
    for i in range(0, len(peaks) - window_len * 500 + 1, 10 * 500):
        temp = peaks[i:i + window_len * 500]
        rri, _, _ = _hrv_format_input(temp, sampling_rate=500)
        res.append(rri)
        time_axis.append(i / 500)

    statis_mu = [norm.fit(r)[0] for r in res]
    statis_sigma = [norm.fit(r)[1] for r in res]
    ecg_stress_flags = [mu < mu_all and sigma <= sigma_all for mu, sigma in zip(statis_mu, statis_sigma)]

    # EEG 处理
    eeg_raw = df['CH2']
    alpha_energies, beta_energies = process_with_sliding_window(eeg_raw, window_size=20, step_size=10, sampling_rate=500)
    _, _, eeg_stress_flags = calculate_dynamic_stress(alpha_energies, beta_energies)

    # EDA 处理
    eda_signal = lowpass_filter(df['CH3'], cutoff=5, fs=500)
    eda_results, _, _, _ = stress_analysis(eda_signal, baseline_mean, baseline_std, sampling_rate=500)
    eda_stress_flags = [res['stress_detected'] for res in eda_results]

    # 综合压力标记
    combined_flags = [sum([ecg, eeg, eda]) >= 2 for ecg, eeg, eda in zip(ecg_stress_flags, eeg_stress_flags, eda_stress_flags)]
    final_stress_flags.append(combined_flags)

    # 计算压力时长与占比
    stress_duration = sum(combined_flags) * 10  # 每段10秒
    total_duration = len(combined_flags) * 10
    stress_percentages.append((stress_duration / total_duration) * 100)
    stress_durations.append(stress_duration)

# 绘制最终压力状态
plt.figure(figsize=(12, 8))
for i, (zone, flags) in enumerate(zip(file_paths.keys(), final_stress_flags)):
    total_time = np.arange(0, len(flags) * 10, 10)
    stress_areas = [10 if flag else 0 for flag in flags]  # 每个压力点的高度设置为10
    ax = plt.subplot(2, 2, i + 1)
    ax.fill_between(total_time, stress_areas, color="red", step="pre", alpha=0.5, label="Stress")
    ax.set_xlabel("Total Time (s)")
    ax.set_ylabel("Stress Level")
    ax.set_title(zone, loc='left')
    ax.legend(loc='upper right')
    ax.grid()

plt.tight_layout()
plt.show()

# 绘制压力占比饼状图
plt.figure(figsize=(12, 8))
for i, (zone, percentage, duration) in enumerate(zip(file_paths.keys(), stress_percentages, stress_durations)):
    ax = plt.subplot(2, 2, i + 1)
    ax.pie([percentage, 100 - percentage], labels=[f"Stress: {percentage:.1f}%", f"Non-Stress: {100 - percentage:.1f}%"], autopct="%1.1f%%")
    ax.set_title(f"{zone} (Total Stress Duration: {duration}s)", loc='left')

plt.tight_layout()
plt.show()
