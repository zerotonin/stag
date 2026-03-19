"""
NaN detection and linear interpolation for sensor data.

Utility functions for locating contiguous NaN sequences in numpy arrays
and filling them via linear interpolation.
"""
import numpy as np
import pandas as pd

def load_data(filename):
    """Loads data from a file. Handles various potential file formats."""
    try:
        # Try loading as a NumPy array directly
        return np.load(filename)
    except ValueError:  
        # File might be CSV or have mixed delimiters; use pandas for flexibility 
        return pd.read_csv(filename, header=None).to_numpy()


def find_nan_sequences(arr):
    """Finds sequences of NaN values within each column."""
    nan_sequences = []
    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]
        is_nan = np.isnan(col)
        if any(is_nan):
            start = None
            for i, val in enumerate(is_nan):
                if val and start is None:
                    start = i
                elif not val and start is not None:
                    nan_sequences.append((col_idx, start, i - 1))
                    start = None
    return nan_sequences


def interpolate_nan_sequences(arr, nan_sequences):
    """Interpolates NaN sequences linearly in each column."""
    for col_idx, start, end in nan_sequences:
        col = arr[:, col_idx]
        y1 = col[start - 1] if start > 0 else np.nan
        y2 = col[end + 1] if end < len(col) - 1 else np.nan
        interp_values = np.linspace(y1, y2, end - start + 1)
        col[start:end + 1] = interp_values


# ----- Main script -----
filename = "/home/geuba03p/deer_accl/clust_data_raw.npy"  # Replace with the name of your data file
data = load_data(filename)

nan_sequences = find_nan_sequences(data)
interpolate_nan_sequences(data, nan_sequences)

# Save the modified data (optional)
np.save("/home/geuba03p/deer_accl/clust_data_noNAN.npy", data)

print("NaN sequences interpolated.")
