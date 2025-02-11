

import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
from sklearn.cluster import KMeans
from neurokit2.hrv.hrv_utils import _hrv_format_input

file_path = r'/Users/liu/Downloads/new/10/opensignals_00078065DFF4_2024-08-28_15-06-58.txt'
df = pd.read_csv(file_path, sep='\s+', comment='#', header=None)
df.columns = ['nSeq', 'DI', 'CH1', 'CH2', 'CH3']  
ecg_raw = df['CH1']
ecg = ecg_raw.tolist()


signals_1, info_1 = nk.ecg_process(ecg, sampling_rate = 500) 
#https://neuropsychology.github.io/NeuroKit/functions/hrv.html
#ecg_raw = signals_1['ECG_Raw'].tolist()
ecg_cleaned = nk.ecg_clean(ecg, sampling_rate = 500)
peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate = 500, correct_artifacts=True)

#https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/hrv/hrv.py
rri, rri_time, rri_missing = _hrv_format_input(peaks, sampling_rate = 500)
mu_all, sigma_all = norm.fit(rri)
#hrv_indices = nk.hrv(peaks, sampling_rate = 500, show=True)
#Histogram
#plt.hist(rri, bins = 30)
#plt.plot(rri)
color_1 = (39/255, 195/255, 243/255)
color_2 = (12/255, 113/255, 133/255)
color_3 = (5/255, 112/255, 145/255)
color_4 = (3/255, 52/255, 83/255)

peaks = peaks.to_numpy()
i = 0 
window_len = 20 #s
res = []
time_axis = []
while (i + window_len * 500 - 1) <= len(peaks):
    start = i
    end = i + window_len * 500 - 1
    temp = peaks[start : end]
    rri, rri_time, rri_missing = _hrv_format_input(temp, sampling_rate = 500)
    res.append(rri)
    time_axis.append(start / 500)  
    i += 5 * 500

mu, sigma = norm.fit(rri)

statis_mu = []
statis_sigma = []
for i in range (len(res)):
    mu, sigma = norm.fit(res[i])
    statis_mu.append(mu)
    statis_sigma.append(sigma)

counter = 0
counter_index = []   
for i in range(len(res)):
    if statis_mu[i] <= mu_all and statis_sigma[i] <= sigma_all:
        counter += 1
        counter_index.append(i)

timer_start = [counter_index[0]]
timer_end = []
for i in range(1, len(counter_index)):
    if counter_index[i] - counter_index[i-1] != 1:
        timer_start.append(counter_index[i])
        timer_end.append(counter_index[i-1])
        
col=[]
for i in range(0,len(statis_mu)):
    if statis_mu[i] >= mu_all:
        if statis_sigma[i] <= sigma_all:
            col.append('g')
        else:
            col.append('b')
    else: 
        if statis_sigma[i] <= sigma_all:
            col.append('r') 
        else:
            col.append('orange') 
        
plt.figure(figsize=(12, 6)) 
plt.xlim(700, 840)  
plt.ylim(20, 80)   
plt.rcParams.update({'font.size': 50})
for i in range(len(statis_mu)):
    plt.scatter(statis_mu[i], statis_sigma[i], s=40, marker='x', c=col[i], linewidth=2)  
# plt.scatter(feature[:,0], feature[:,1])
plt.axvline(x=mu_all, color='b', linewidth=2, linestyle='--')
plt.axhline(y=sigma_all, color='b', linewidth=2, linestyle='--')
plt.grid(True)
plt.xlabel('Mean of RRI (ms)')
plt.ylabel('Standard Deviation of RRI (ms)')
mpl.rcParams.update({'font.size': 14})

time_axis_seconds = [t for t in time_axis]  
plt.figure(figsize=(12, 6))
plt.plot(time_axis_seconds, statis_mu, label='RRI Mean (μ)', color='blue', linewidth=2)
plt.plot(time_axis_seconds, statis_sigma, label='RRI Std (σ)', color='green', linewidth=2)

for i, color in enumerate(col):
    if color == 'r':
        plt.axvspan(time_axis_seconds[i], time_axis_seconds[i] + window_len, color='red', alpha=0.1, label='High Stress' if i == 0 else "")

plt.xlabel('Time (s)')
plt.ylabel('Feature Value')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6)) 

plt.axvline(x=mu_all, color='b', linewidth=8, linestyle='--')
plt.axhline(y=sigma_all, color='b', linewidth=8, linestyle='--')
plt.xlabel('Mean of RRI (ms)')
plt.ylabel('Standard Deviation of RRI (ms)')
mpl.rcParams.update({'font.size': 14})
