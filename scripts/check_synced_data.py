import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

paths = {
    "alex_paths": {
        "rawdata_folder": r'C:\Users\Lindsay\Documents\alexander\Deer_project\deer_code\all_raw_data',
        "deer_code_filepath": r'C:\Users\Lindsay\Documents\alexander\Deer_project\deer_code\Deer_codes.csv',
        "merged_signal_file": r'C:\Users\Lindsay\Documents\alexander\Deer_project\deer_code\normalised_calibration\data_out',
        "plot_file": r'C:\Users\Lindsay\Documents\alexander\Deer_project\deer_code\normalised_calibration\plots_nc',
        "log_file": r"C:\Users\Lindsay\Documents\alexander\Deer_project\deer_code\normalised_calibration\logs"
    },
    "bart_paths": {
        # Bart's paths
        "rawdata_folder": r'',
        "deer_code_filepath": r'',
        "merged_signal_file": r'',
        "plot_file": r'',
        "log_file": r""
    },
    "cluster_paths": {
        # Cluster's paths
        "rawdata_folder": r'/projects/sciences/zoology/geurten_lab/files_extracted/raw_data/',
        "deer_code_filepath": r'/projects/sciences/zoology/geurten_lab/files_extracted/Deer_codes.csv',
        "merged_signal_file": r'/projects/sciences/zoology/geurten_lab/files_extracted/sync_file_results/data/',
        "plot_file": r'/projects/sciences/zoology/geurten_lab/files_extracted/sync_file_results/plots/',
        "log_file": r"/projects/sciences/zoology/geurten_lab/files_extracted/sync_file_results/logs/"
    },
    "cluster_paths_2": {
        # Cluster's paths
        "rawdata_folder": r'/projects/sciences/zoology/geurten_lab/files_extracted/raw_data/',
        "deer_code_filepath": r'/projects/sciences/zoology/geurten_lab/files_extracted/Deer_codes.csv',
        "merged_signal_file": r'/projects/sciences/zoology/geurten_lab/files_extracted/new_file_sync/data/',
        "plot_file": r'/projects/sciences/zoology/geurten_lab/files_extracted/new_file_sync/plots/',
        "log_file": r"//projects/sciences/zoology/geurten_lab/files_extracted/new_file_sync/logs/"
    }
    # You can add more path sets as needed
}
path_sys="cluster_paths_2"

rawdata_folder     = paths[path_sys]['rawdata_folder']
deer_code_filepath = paths[path_sys]['deer_code_filepath']
merged_signal_file = paths[path_sys]['merged_signal_file']
plot_file          = paths[path_sys]['plot_file']
log_file           = paths[path_sys]['log_file']
# Define the directory where your h5 files are located
directory = merged_signal_file

# Define the directory to save the plots
save_directory = r'/projects/sciences/zoology/geurten_lab/files_extracted/new_file_sync/quality_control/'


correlation_file = r'/projects/sciences/zoology/geurten_lab/files_extracted/new_file_sync/quality_control/correlations.txt'
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
            all_accell[['X_head', 'Y_head', 'Z_head']] = (all_accell[['X_head', 'Y_head', 'Z_head']] - all_accell[['X_head', 'Y_head', 'Z_head']].mean()) / all_accell[['X_head', 'Y_head', 'Z_head']].std()
            all_accell[['X_ear', 'Y_ear', 'Z_ear']] = (all_accell[['X_ear', 'Y_ear', 'Z_ear']] - all_accell[['X_ear', 'Y_ear', 'Z_ear']].mean()) / all_accell[['X_ear', 'Y_ear', 'Z_ear']].std()
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