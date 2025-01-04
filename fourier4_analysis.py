import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

# Deep sleep data (in minutes per day)
deep_sleep_data = np.array([
    61, 70, 77, 87, 56, 53, 78, 63, 47, 104, 47, 87, 74, 77, 65, 39, 69, 83, 44, 42, 79,
    81, 63, 49, 90, 107, 102, 68, 76, 52, 78, 77, 63, 72, 76, 68, 100, 71, 85, 77, 96, 105,
    86, 109, 72, 72, 84, 84, 106, 72, 37, 83, 82, 98, 117, 61, 95, 49, 68, 83, 80, 85, 63, 
    66, 101, np.nan, 77, 61, 91, 98, 70, np.nan, 79, 67, 84, 56, 64, 62, 64, 90, 87, np.nan,
    76, 81, 82, 43, 115, 41, 55, 58, 83, 68, 57, 84, 73, 70, 59, 85, 81, 65, 99, 95, 55, 
    101, 73, 80, 41, 83, 70, 79, 78, 80, 76, 84, 63, 68, 47
])

# Remove #N/A or NaN values by ignoring them in the FFT calculation
cleaned_data = deep_sleep_data[~np.isnan(deep_sleep_data)]

# Detrend the data to remove any long-term trends
detrended_data = detrend(cleaned_data)

# Perform FFT (Fast Fourier Transform)
fft_result = np.fft.fft(detrended_data)

# Frequency axis: length of data is the number of days
num_days = len(detrended_data)
frequencies = np.fft.fftfreq(num_days)

# We are only interested in the positive frequencies (real part of the FFT)
positive_frequencies = frequencies[:num_days // 2]
fft_magnitude = np.abs(fft_result)[:num_days // 2]

# Convert frequencies to periods (in days)
periods_in_days = 1 / positive_frequencies

# Set up plot
plt.figure(figsize=(10, 6))
plt.plot(periods_in_days, fft_magnitude, color='orange')
plt.xlabel("Period (days)", fontsize=12)
plt.ylabel("FFT Magnitude", fontsize=12)
plt.title("Frequency Spectrum of Detrended Deep Sleep Data", fontsize=14)
plt.xlim(0, 15)  # Show up to 100 days in the period range (for monthly trends)
plt.yscale('log')  # Log scale to see all magnitudes
plt.grid(True)
plt.show()