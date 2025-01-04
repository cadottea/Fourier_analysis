import numpy as np
import pandas as pd

# Your actual deep sleep data with missing values represented as #N/A
sleep_data = [
    61, 70, 77, 87, 56, 53, 78, 63, 47, 104, 47, 87, 74, 77, 65, 39, 69, 83, 44, 42, 79, 
    81, 63, 49, 90, 107, 102, 68, 76, 52, 78, 77, 63, 72, 76, 68, 100, 71, 85, 77, 96, 105,
    86, 109, 72, 72, 84, 84, 106, 72, 37, 83, 82, 98, 117, 61, 95, 49, 68, 83, 80, 85, 63, 
    66, 101, np.nan, 77, 61, 91, 98, 70, np.nan, 79, 67, 84, 56, 64, 62, 64, 90, 87, np.nan, 
    76, 81, 82, 43, 115, 41, 55, 58, 83, 68, 57, 84, 73, 70, 59, 85, 81, 65, 99, 95, 55, 
    101, 73, 80, 41, 83, 70, 79, 78, 80, 76, 84, 63, 68, 47
]

# Step 1: Replace #N/A (np.nan) with the average of surrounding values (mean of non-NaN values)
# Convert np.nan to None (used for missing data in NumPy)
sleep_data = [None if np.isnan(x) else x for x in sleep_data]

# Fill the missing values (None) with the average of the surrounding values
for i in range(1, len(sleep_data)-1):
    if sleep_data[i] is None:
        # Average of the previous and next value
        sleep_data[i] = (sleep_data[i-1] + sleep_data[i+1]) / 2

# Convert the data to a NumPy array
sleep_data = np.array(sleep_data)

# Step 2: Perform Fast Fourier Transform (FFT)
fft_result = np.fft.fft(sleep_data)

# Get the magnitude of the FFT (absolute value)
fft_magnitude = np.abs(fft_result)

# Step 3: Create the corresponding frequency axis
n = len(sleep_data)
sampling_rate = 1  # Assuming 1 data point per day
frequencies = np.fft.fftfreq(n, d=sampling_rate)

# Step 4: Combine frequencies and magnitudes into a DataFrame
df = pd.DataFrame({
    'Frequency': frequencies[:n//2],  # Only use the positive frequencies
    'FFT Magnitude': fft_magnitude[:n//2]  # Only use the positive magnitudes
})

# Step 5: Save to CSV file for Excel plotting
df.to_csv("decomposition_with_frequencies.csv", index=False)

print("Fourier decomposition with frequencies saved to 'decomposition_with_frequencies.csv'.")