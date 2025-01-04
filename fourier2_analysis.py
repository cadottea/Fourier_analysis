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