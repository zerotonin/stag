import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stag.local_paths import get_path

# Every path was previously held in a multi-machine dictionary
# (alex_paths / bart_paths / cluster_paths / cluster_paths_2) with
# absolute Windows + Linux paths hardcoded inline.  All of that is
# now in local_paths.json - one source of truth per machine.
rawdata_folder     = get_path("aoraki_raw_data")
deer_code_filepath = get_path("aoraki_deer_codes")
merged_signal_file = get_path("aoraki_merged_signals_v2")
plot_file          = get_path("aoraki_plot_dir_v2")
log_file           = get_path("aoraki_log_dir_v2")

# Define the directory where your h5 files are located
directory = merged_signal_file

# Quality-control output directory + correlations file.
save_directory   = get_path("aoraki_quality_control_dir")
correlation_file = get_path("aoraki_correlations_file")
# Function to extract R and D numbers from filenames
def extract_R_and_D(filename):
    parts = filename.split('_')
    R = parts[0][1:]
    D = parts[1][1:]
    return f'R{R}D{D}'

def calculate_correlation(series1, series2):
    return np.corrcoef(series1, series2)[0, 1]

with open(correlation_file, 'w') as f:
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.h5'):
            filepath = os.path.join(directory, filename)
            key = "df"  # Assuming the key to read from is always "df"

            # Read the h5 file
            all_accell = pd.read_hdf(filepath, key=key)
            head_cols = ["X_head", "Y_head", "Z_head"]
            ear_cols  = ["X_ear", "Y_ear", "Z_ear"]
            all_accell[head_cols] = (
                (all_accell[head_cols] - all_accell[head_cols].mean())
                / all_accell[head_cols].std()
            )
            all_accell[ear_cols] = (
                (all_accell[ear_cols] - all_accell[ear_cols].mean())
                / all_accell[ear_cols].std()
            )
            # Calculate the sum of accelerometer data for the first 20000 values
            sum_head_first_20000 = all_accell[['X_head', 'Y_head', 'Z_head']].iloc[:20000].sum(axis=1)
            sum_ear_first_20000 = all_accell[['X_ear', 'Y_ear', 'Z_ear']].iloc[:20000].sum(axis=1)

            # Plotting the sum of accelerometer data for the first 20000 values
            plt.figure(figsize=(12, 6))
            plt.plot(all_accell['NZ_DateTime'].iloc[:20000], sum_head_first_20000, label='Head Sum')
            plt.plot(all_accell['NZ_DateTime'].iloc[:20000], sum_ear_first_20000, label='Ear Sum')
            plt.xlabel('NZ_DateTime')
            plt.ylabel('Sum of Accelerometer Readings')
            plt.title('Sum of Accelerometer Readings (First 20000 Values)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the start plot
            plot_filename = extract_R_and_D(filename) + '_start.png'
            save_path = os.path.join(save_directory, plot_filename)
            plt.savefig(save_path)
            plt.close()

            # Calculate the sum of accelerometer data for the last 20000 values
            sum_head_last_20000 = all_accell[['X_head', 'Y_head', 'Z_head']].iloc[-20000:].sum(axis=1)
            sum_ear_last_20000 = all_accell[['X_ear', 'Y_ear', 'Z_ear']].iloc[-20000:].sum(axis=1)

            # Plotting the sum of accelerometer data for the last 20000 values
            plt.figure(figsize=(12, 6))
            plt.plot(all_accell['NZ_DateTime'].iloc[-20000:], sum_head_last_20000, label='Head Sum')
            plt.plot(all_accell['NZ_DateTime'].iloc[-20000:], sum_ear_last_20000, label='Ear Sum')
            plt.xlabel('NZ_DateTime')
            plt.ylabel('Sum of Accelerometer Readings')
            plt.title('Sum of Accelerometer Readings (Last 20000 Values)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the end plot
            plot_filename = extract_R_and_D(filename) + '_end.png'
            save_path = os.path.join(save_directory, plot_filename)
            plt.savefig(save_path)
            plt.close()

            # Calculate the sum of accelerometer data for the middle 20000 values
            middle_start = len(all_accell) // 2 - 10000
            middle_end = len(all_accell) // 2 + 10000
            sum_head_middle_20000 = all_accell[['X_head', 'Y_head', 'Z_head']].iloc[middle_start:middle_end].sum(axis=1)
            sum_ear_middle_20000 = all_accell[['X_ear', 'Y_ear', 'Z_ear']].iloc[middle_start:middle_end].sum(axis=1)

            # Plotting the sum of accelerometer data for the middle 20000 values
            plt.figure(figsize=(12, 6))
            plt.plot(all_accell['NZ_DateTime'].iloc[middle_start:middle_end], sum_head_middle_20000, label='Head Sum')
            plt.plot(all_accell['NZ_DateTime'].iloc[middle_start:middle_end], sum_ear_middle_20000, label='Ear Sum')
            plt.xlabel('NZ_DateTime')
            plt.ylabel('Sum of Accelerometer Readings')
            plt.title('Sum of Accelerometer Readings (Middle 20000 Values)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the middle plot
            plot_filename = extract_R_and_D(filename) + '_middle.png'
            save_path = os.path.join(save_directory, plot_filename)
            plt.savefig(save_path)
            plt.close()


            # Compute the sum of accelerations for each accelerometer
            sum_head = all_accell[['X_head', 'Y_head', 'Z_head']].sum(axis=1)
            sum_ear = all_accell[['X_ear', 'Y_ear', 'Z_ear']].sum(axis=1)

            # Compute cross-correlation between summed accelerations
            corr = np.correlate(sum_ear, sum_head, mode='full')

            # Calculate the lag corresponding to the maximum correlation
            lag = np.argmax(corr) - len(sum_ear) + 1

            R_D_name = extract_R_and_D(filename)  # You need to define this function
            f.write(f"{R_D_name} : Correlation: {calculate_correlation(sum_head, sum_ear)}, Lag: {lag}\n")
