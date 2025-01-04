Fourier Analysis for Sleep Trend Detection

This repository contains Python code for performing Fourier analysis on data to detect trends, specifically for analyzing sleep patterns using FitBit data. However, the techniques can be applied to any signal analysis task that benefits from Fourier transforms.

Features
    •   Fourier Analysis Options: Includes Savitzky-Golay, Gaussian, Low-pass filter, and Band-pass filter.
    •   Sleep Trend Analysis: Designed for analyzing sleep data, particularly Deep Sleep from FitBit data, but applicable to other time series data.
    •   Efficient Signal Processing: Provides various filtering and smoothing techniques to help identify patterns in noisy data.

Prerequisites

To use this repository, you’ll need to have Python 3.6 or later installed, along with the following dependencies:
    •   NumPy
    •   SciPy
    •   Matplotlib

Installation
    1.  Clone the repository to your local machine:
git clone https://github.com/cadottea/FOURIER_ANALYSIS.git
    2.  Install the required dependencies:
pip install -r requirements.txt

Fourier Analysis Techniques

This project provides different Fourier analysis techniques for signal processing, which include:
    1.  Savitzky-Golay Filtering: A smoothing method that applies a polynomial filter to the signal. This is useful for reducing noise in data with small fluctuations.
    2.  Gaussian Filter: A Gaussian filter can be used to smooth the data by applying a weighted average based on the Gaussian distribution, effectively reducing high-frequency noise.
    3.  Low-pass Filter: This filter allows frequencies below a certain cutoff to pass through, and attenuates higher frequencies. It’s useful when you want to preserve the low-frequency components of a signal (such as slow changes in sleep patterns).
    4.  Band-pass Filter: A band-pass filter is designed to pass frequencies within a certain range and attenuate frequencies outside this range. It is particularly useful when you want to isolate signals in a specific frequency band.

Example Usage

Assuming you have Deep Sleep data (which could be in a list or numpy array, including NaNs), you can use the following example code to apply Fourier analysis and perform signal processing. This includes replacing NaNs, applying a window function, performing the FFT, and plotting the results. You would start by loading the Deep Sleep data as a numpy array, replacing NaNs with the average of the available values, and then applying the Fourier analysis:

import numpy as np
import matplotlib.pyplot as plt

# Assuming 'data' is your Deep Sleep data as a list or numpy array (including NaNs)
# Replace NaNs with the average of the available values to deal with missing data
data = np.array([61, 70, 77, 87, 56, 53, 78, 63, 47, 104, 47, 87, 74, 77, 65, 39, 69, 83, 44, 42, 79, 81, 63, 49, 
                 90, 107, 102, 68, 76, 52, 78, 77, 63, 72, 76, 68, 100, 71, 85, 77, 96, 105, 86, 109, 72, 72, 84, 84, 
                 106, 72, 37, 83, 82, 98, 117, 61, 95, 49, 68, 83, 80, 85, 63, 66, 101, np.nan, 77, 61, 91, 98, 70, np.nan,
                 79, 67, 84, 56, 64, 62, 64, 90, 87, np.nan, 76, 81, 82, 43, 115, 41, 55, 58, 83, 68, 57, 84, 73, 70, 59, 
                 85, 81, 65, 99, 95, 55, 101, 73, 80, 41, 83, 70, 79, 78, 80, 76, 84, 63, 68, 47])

# Replace NaNs with the average of the available values
data_no_nans = np.nan_to_num(data, nan=np.nanmean(data))

# Apply Hanning window to reduce spectral leakage
window = np.hanning(len(data_no_nans))
data_windowed = data_no_nans * window

# Perform FFT
fft_result = np.fft.fft(data_windowed)
fft_magnitude = np.abs(fft_result)  # Magnitude of the FFT result
frequencies = np.fft.fftfreq(len(data_no_nans))  # Frequencies corresponding to the FFT

# Only consider the positive frequencies (half of the FFT result)
positive_freqs = frequencies[:len(frequencies)//2]
positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]

# Plot the FFT result
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, positive_magnitude)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Magnitude of Deep Sleep Data (Windowed)")
plt.grid(True)
plt.show()

Conclusion

This repository provides tools for efficient signal processing and Fourier analysis that can be applied to a variety of signal types, including sleep trend analysis from FitBit data. With the ability to filter and analyze frequency components, this tool is powerful for detecting trends and patterns within noisy data.