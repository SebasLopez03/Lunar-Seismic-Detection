# LUNAR SEISMIC DETECTOR | NASA SPACE APPS 2024 | 
# The following code takes a MSEED file and marks the arrival time of the seismic event with highest magnitude, if any.
# It uses different filters to better describe the signals and a STA/LTA algorithm to find the arrival time of each sample.
# Finally, it stores the arrival time of each file in a catalog in a CSV file.

# Import libraries
import numpy as np
import pandas as pd
import os

from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frecuency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

#  The following function takes the absolute values of a data list-alike object and uses two filters to smooth-out the data.
def smooth_data(data):
    data_abs = []
    for point in data:
        data_abs.append(abs(point))
        
    data_lowpass = lowpass_filter(data_abs, cutoff, fs)
    data_savgol = savgol_filter(data_lowpass, 1000, 2)
    
    return data_savgol

# Similarly, 
def smooth_stream(stream):
    # Set the minimum frequency
    minfreq = 0.1
    maxfreq = 1

    # Going to create a separate trace for the filter data
    st_filt = stream.copy()
    st_filt.filter('bandpass', freqmin=minfreq ,freqmax=maxfreq)
    tr_filt = st_filt.traces[0].copy()
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data
    
    tr_data_abs = []
    for point in tr_data_filt:
        tr_data_abs.append(abs(point))
        
    tr_data_lowpass = lowpass_filter(tr_data_abs, cutoff, fs)
    tr_data_savgol = savgol_filter(tr_data_lowpass, window_size, 3)
    
    return tr_times_filt, tr_data_savgol

# The function iterates over a data array and calculates the acumulative sum in each step. Graphing the result can help identify 
# seismic events, which usually have a sigmoid like aspect.
def squared_sumation(data):
    sumation = 0
    array = []
    for point in data:
        sumation = sumation + point**2
        array.append(sumation)
        
    return np.array(array)

# We chose to generate some random noise at the beginning of each sample to simulate a continuous signal input and to allow
# bigger Long Term windows in the STA/LTA algorithm. 
def generate_initial_data(length, data, noise_level=1e-10):
    initial_data = np.random.normal(loc=0, scale=noise_level, size=length)
    initial_data += np.mean(data[:length])
    return initial_data

# In this case, we took the S12 Grade Lunar dataset to test our algorithm.
# Make sure to change the relative path when running the code. 
dir = "src\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\lunar/test\data\S12_GradeB/"
list_dir = os.listdir(dir)

# Filter parameters
# Savitzky-Golay filter
window_size = 300 # Must be odd
poly_order = 3 # Polynomial order
# Lowpass filter
cutoff = 0.1  # Cut frequency
fs = 500.0  #  Signal sampling frequency

if __name__ == '__main__':
    
    detection = False
    detection_times = []
    fnames = []
    on_off = np.array([])
    
    for filename in list_dir:
        try: 
            print(f"Progress: {(list_dir.index(filename)/len(list_dir) * 100)}%")
        except:
            print("Complete! 100%")
            
        if filename[-5:] == 'mseed':
            # print(filename)
            stream = read(dir + filename)
            trace = stream[0]
            
            tr_times = trace.times()
            tr_data = trace.data
            starttime = trace.stats.starttime.datetime
            
            # Sampling frequency of our trace
            df = trace.stats.sampling_rate
            
            # Random initial data
            initial_length = 10000
            initial_data = generate_initial_data(initial_length, tr_data)
            initial_times = np.linspace(-initial_length * df, 0, initial_length)
            
            # Combine initial data with real data
            tr_times = np.concatenate([initial_times, tr_times])
            tr_data = np.concatenate([initial_data, tr_data])
            
            tr_times_filt, tr_data_smooth = smooth_stream(stream)
            
            max_smooth = max(tr_data_smooth)
            time_max = tr_times_filt[np.where(tr_data_smooth == max_smooth)]  
            
            # Set the minimum frequency
            minfreq = 0.1
            maxfreq = 1
            
            # First data filters
            st_filt = stream.copy()
            st_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
            tr_filt = st_filt.traces[0].copy()
            tr_data_filt = tr_filt.data
            tr_times_filt = tr_filt.times()
            
            # Adjust filtered data
            tr_times_filt = np.concatenate([tr_times_filt, tr_times_filt])
            tr_data_smooth = np.concatenate([tr_data_smooth, tr_data_smooth])
            tr_data_filt = np.concatenate([tr_data_filt, tr_data_filt])
            
            # STA/LTA algorithm to find the characteristic function
            cft = classic_sta_lta(tr_data_filt, int(700 * df), int(9000 * df))
            
            # These thresholds values may vary depending on the dataset
            thr_on = 2
            thr_off = 1.5
            
            on_off = np.array(trigger_onset(cft, thr_on, thr_off))
            
            fig, axs = plt.subplots(3, 1)
            
            # To avoid several detections in a single sample or false detections, the maximum point of the smoothed curve must fall
            # in a given interval after the suposed arrival time. 
            for i in np.arange(0, len(on_off)):
                triggers = on_off[i]
                
                interval = tr_times_filt[triggers[0]-1000:triggers[0] + 5000]
                
                if time_max in interval:
                    # Real event
                    axs[0].axvline(x=tr_times_filt[triggers[0]], color='red', label='Trig. On')
                    axs[0].axvline(x=tr_times_filt[triggers[1]], color='orange', label='Trig. Off')
                    detection = True
                else:
                    # False detection
                    axs[0].axvline(x=tr_times_filt[triggers[0]], color='cyan', label='Trig. On')
                    axs[0].axvline(x=tr_times_filt[triggers[1]], color='blue', label='Trig. Off')
            
            axs[0].axvline(x = time_max, color='magenta', label='Time_max')
            axs[0].plot(tr_times[initial_length:], tr_data[initial_length:])
            axs[0].plot(tr_times_filt[initial_length:], tr_data_smooth[initial_length:])
            axs[1].plot(tr_times_filt, cft)
            axs[2].plot(tr_times, squared_sumation(tr_data))
            fig.savefig(f'src\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\lunar/test\plots\S12_GradeB/{filename[:-4]}.png')
            # plt.show()
            plt.close()
            
            if detection:
                on_time = starttime + timedelta(seconds = tr_times_filt[triggers[0]])
                on_time_str = datetime.strftime(on_time,'%Y-%m-%dT%H:%M:%S.%f')
                detection_times.append(on_time_str)
            else:
                detection_times.append('-')
                
            fnames.append(filename)
            
    # Compile dataframe of detections
    print(len(fnames))
    print(len(detection_times))
    
    detect_df = pd.DataFrame(data = {'filename':fnames, 'time_abs':detection_times})
    detect_df.head()
    detect_df.to_csv(f'src\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\lunar/test\catalogs\S12_GradeB/catalog.csv')