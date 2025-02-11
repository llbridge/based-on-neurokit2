import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt

# File paths for datasets
file_paths = {
    "a) Still Zone": r'/Users/liu/Downloads/new/15/opensignals_00078065DFF4_2024-09-02_16-23-17.txt',
    "b) 30-zone": r'/Users/liu/Downloads/new/15/opensignals_00078065DFF4_2024-09-02_16-28-29.txt',
    "c) 50-zone": r'/Users/liu/Downloads/new/15/opensignals_00078065DFF4_2024-09-02_16-39-24.txt',
    "d) Highway": r'/Users/liu/Downloads/new/15/opensignals_00078065DFF4_2024-09-02_16-53-02.txt'
}

# Function for wavelet decomposition to extract Alpha and Beta energies
def wavelet_decomposition_alpha_beta(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    alpha_coeffs = coeffs[level - 2]  # Corresponds to 8-13 Hz
    beta_coeffs = coeffs[level - 3]  # Corresponds to 13-30 Hz
    alpha_energy = np.sum(np.square(alpha_coeffs))
    beta_energy = np.sum(np.square(beta_coeffs))
    return alpha_energy, beta_energy

# Sliding window processing
def process_with_sliding_window(signal, window_size, step_size, sampling_rate, wavelet='db4', level=6):
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

# Dynamic threshold calculation and stress detection
def calculate_dynamic_stress(alpha_energies, beta_energies, window_size=20, k=1):
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

        # Stress detection
        stress_indices.append(ratios[i] > threshold)

    return ratios, thresholds, stress_indices

# High stress duration filtering
def filter_stress_windows(is_stress, min_duration=1):
    """
    Filters high-stress windows, ensuring the duration exceeds the minimum threshold.
    """
    filtered_stress = [False] * len(is_stress)
    count = 0
    for i in range(len(is_stress)):
        if is_stress[i]:
            count += 1
        else:
            if count >= min_duration:
                for j in range(i - count, i):
                    filtered_stress[j] = True
            count = 0
    if count >= min_duration:
        for j in range(len(is_stress) - count, len(is_stress)):
            filtered_stress[j] = True
    return filtered_stress

# Process each dataset
results = {}
for key, path in file_paths.items():
    df = pd.read_csv(path, sep='\s+', comment='#', header=None)
    df.columns = ['nSeq', 'DI', 'CH1', 'CH2', 'CH3']
    eeg_raw = df['CH2']
    results[key] = process_with_sliding_window(eeg_raw, window_size=20, step_size=10, sampling_rate=500)

# Calculate stress and apply dynamic thresholding
stress_results = {}
for zone, (alpha_energies, beta_energies) in results.items():
    ratios, thresholds, is_stress = calculate_dynamic_stress(alpha_energies, beta_energies)
    filtered_is_stress = filter_stress_windows(is_stress, min_duration=1)
    stress_results[zone] = {
        "ratios": ratios,
        "thresholds": thresholds,
        "is_stress": is_stress,
        "filtered_is_stress": filtered_is_stress
    }

# Plot Stress Index and Dynamic Threshold for all zones on a single canvas
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (zone, stress_data) in enumerate(stress_results.items()):
    total_time = np.arange(0, len(stress_data["ratios"]) * 5, 5)  # Total time in seconds
    axes[i].plot(total_time, stress_data["ratios"], label=f'Stress Index', marker='o')
    axes[i].plot(total_time, stress_data["thresholds"], label=f'Dynamic Threshold', linestyle='--')

    # Add red shaded regions for high stress
    for j, is_stress in enumerate(stress_data["filtered_is_stress"]):
        if is_stress:
            axes[i].axvspan(j * 5, (j + 1) * 5, color='red', alpha=0.5)

    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Beta/Alpha')
    axes[i].legend(loc='upper right')
    axes[i].grid(True)
    axes[i].set_title(f'{zone}',loc='left')
plt.tight_layout()
plt.show()

# Print filtered high-stress windows and calculate total stress duration
for zone, stress_data in stress_results.items():
    filtered_stress_windows = [i for i, is_stress in enumerate(stress_data["filtered_is_stress"]) if is_stress]
    total_stress_duration = len(filtered_stress_windows) * 10  
    print(f"{zone}: Filtered high stress windows {filtered_stress_windows}")
    print(f"{zone}: Total stress duration = {total_stress_duration} seconds")
