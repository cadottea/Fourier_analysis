import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Deep sleep data (in minutes per day)
deep_sleep_data = np.array([
    61, 70, 77, 87, 56, 53, 78, 63, 47, 104, 47, 87, 74, 77, 65, 39, 69, 83, 44, 42, 79,
    81, 63, 49, 90, 107, 102, 68, 76, 52, 78, 77, 63, 72, 76, 68, 100, 71, 85, 77, 96, 105,
    86, 109, 72, 72, 84, 84, 106, 72, 37, 83, 82, 98, 117, 61, 95, 49, 68, 83, 80, 85, 63, 
    66, 101, np.nan, 77, 61, 91, 98, 70, np.nan, 79, 67, 84, 56, 64, 62, 64, 90, 87, np.nan,
    76, 81, 82, 43, 115, 41, 55, 58, 83, 68, 57, 84, 73, 70, 59, 85, 81, 65, 99, 95, 55, 
    101, 73, 80, 41, 83, 70, 79, 78, 80, 76, 84, 63, 68, 47
])

# Remove NaN values for the FFT calculation
cleaned_data = deep_sleep_data[~np.isnan(deep_sleep_data)]

# Get days as x-axis
x_vals = np.arange(len(cleaned_data))  # Days

# Polynomial fit of degree 6 (detrending the data)
coeffs = np.polyfit(x_vals, cleaned_data, 6)
polynomial_fit = np.polyval(coeffs, x_vals)

# Detrend the data by subtracting the polynomial fit
detrended_data = cleaned_data - polynomial_fit

# Apply Savitzky-Golay filter for smoothing (window length of 7 and polynomial order of 3)
smoothed_data = savgol_filter(detrended_data, window_length=7, polyorder=3)

# Perform FFT (Fast Fourier Transform) on smoothed data
fft_result = np.fft.fft(smoothed_data)

# Frequency axis: length of data is the number of days
num_days = len(smoothed_data)
frequencies = np.fft.fftfreq(num_days)

# We are only interested in the positive frequencies (real part of the FFT)
positive_frequencies = frequencies[:num_days // 2]
fft_magnitude = np.abs(fft_result)[:num_days // 2]

# Convert frequencies to periods (in days)
periods_in_days = 1 / positive_frequencies

# Set up the figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# First subplot: Original data vs. Smoothed data
axes[0].plot(x_vals, cleaned_data, label="Original Data", color='blue', alpha=0.6)
axes[0].plot(x_vals, smoothed_data, label="Smoothed Data (Savitzky-Golay)", color='orange', alpha=0.9)
axes[0].set_title("Original vs. Smoothed Deep Sleep Data", fontsize=14)
axes[0].set_xlabel("Day", fontsize=12)
axes[0].set_ylabel("Deep Sleep (minutes)", fontsize=12)
axes[0].legend(loc='upper right')
axes[0].grid(True)

# Second subplot: FFT of the smoothed data
axes[1].plot(periods_in_days, fft_magnitude, color='orange')
axes[1].set_title("Frequency Spectrum of Smoothed Deep Sleep Data (Savitzky-Golay)", fontsize=14)
axes[1].set_xlabel("Period (days)", fontsize=12)
axes[1].set_ylabel("FFT Magnitude", fontsize=12)
axes[1].set_xlim(0, 100)  # Show up to 100 days in the period range (for monthly trends)
axes[1].set_yscale('log')  # Log scale to see all magnitudes
axes[1].grid(True)

plt.tight_layout()  # Adjust subplots to fit neatly
plt.show()