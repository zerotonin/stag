# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — analysis.nan_handler                                     ║
# ║  « interpolate or drop NaN gaps »                                ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Utility for repairing the feature matrix prior to               ║
# ║  clustering.  See the docstring for the gap-size cutoff.         ║
# ╚══════════════════════════════════════════════════════════════════╝
"""NaN detection and linear interpolation for sensor data."""

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interpolate NaN runs in a raw accelerometer .npy array.",
    )
    parser.add_argument("infile", help="Input .npy (or CSV) feature matrix.")
    parser.add_argument("outfile", help="Output .npy with NaN runs interpolated.")
    args = parser.parse_args()

    data = load_data(args.infile)
    interpolate_nan_sequences(data, find_nan_sequences(data))
    np.save(args.outfile, data)
    print("NaN sequences interpolated.")
