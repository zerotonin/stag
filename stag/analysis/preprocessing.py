"""
Feature-level statistics for z-scoring.

Computes per-column mean (mu) and standard deviation (sigma) from a
``.npy`` feature matrix and saves them to CSV for use in standardisation
and de-standardisation of cluster centroids.
"""
import numpy as np
import pandas as pd

def save_mu_sigma_from_npy(npy_file_path, csv_file_path):
    """
    Loads a matrix from a .npy file, calculates mu and sigma for each column,
    and saves these values to a CSV file.

    Parameters:
    - npy_file_path (str): Path to the .npy file containing the matrix.
    - csv_file_path (str): Path to save the CSV file with mu and sigma values.
    """
    # Load the matrix from the .npy file
    data_matrix = np.load(npy_file_path)

    # Calculate mu (mean) and sigma (standard deviation) for each column
    mu = np.mean(data_matrix, axis=0)
    sigma = np.std(data_matrix, axis=0)

    # Create a DataFrame with mu and sigma
    stats_df = pd.DataFrame({'mu': mu, 'sigma': sigma})

    # Save the DataFrame to a CSV file
    stats_df.to_csv(csv_file_path, index=False)

    print(f"Mu and sigma values saved to {csv_file_path}")
