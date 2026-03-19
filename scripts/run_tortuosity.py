import pandas as pd
import os
import argparse
import extract_tortuosity
def load_and_clean_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Data file {filepath} does not exist.")
        return pd.DataFrame([])
    data=pd.read_hdf(filepath)
    return data


def process_data(deer_code, path_sys):

    paths = {
    "alex_paths": {
        "rawdata_folder": r'C:\Users\Lindsay\Documents\alexander\Deer_project\data\all_raw_data',
        "deer_code_filepath": r'C:\Users\Lindsay\Documents\alexander\Deer_project\deer_code\Deer_codes_problem.csv',
        "merged_signal_file": r'C:\Users\Lindsay\Documents\alexander\Deer_project\data\normalised\data_out',
        "plot_file": r'C:\Users\Lindsay\Documents\alexander\Deer_project\data\normalised\plots_nc',
        "log_file": r"C:\Users\Lindsay\Documents\alexander\Deer_project\data\normalised\logs",
        "saved_loc_and_tort_filepath": ""
        },
    "bart_paths": {
        # Bart's paths
        "rawdata_folder": r'',
        "deer_code_filepath": r'',
        "merged_signal_file": r'',
        "plot_file": r'',
        "log_file": r"",
        "saved_loc_and_tort_filepath": ""
    },
    "cluster_paths": {
        # Cluster's paths
        "rawdata_folder": r'/projects/sciences/zoology/geurten_lab/files_extracted/raw_data/',
        "deer_code_filepath": r'/projects/sciences/zoology/geurten_lab/files_extracted/Deer_codes.csv',
        "merged_signal_file": r'/projects/sciences/zoology/geurten_lab/files_extracted/sync_file_results/data/',
        "plot_file": r'/projects/sciences/zoology/geurten_lab/files_extracted/sync_file_results/plots/',
        "log_file": r"/projects/sciences/zoology/geurten_lab/files_extracted/sync_file_results/logs/",
        "saved_loc_and_tort_filepath":""
    },
    "cluster_paths_2": {
        # Cluster's paths
        "rawdata_folder": r'/projects/sciences/zoology/geurten_lab/files_extracted/raw_data/',
        "deer_code_filepath": r'/projects/sciences/zoology/geurten_lab/files_extracted/Deer_codes.csv',
        "merged_signal_file": r'/projects/sciences/zoology/geurten_lab/files_extracted/new_file_sync/data/',
        "plot_file": r'/projects/sciences/zoology/geurten_lab/files_extracted/new_file_sync/plots/',
        "log_file": r"/projects/sciences/zoology/geurten_lab/files_extracted/new_file_sync/logs/",
        "saved_loc_and_tort_filepath": r"/projects/sciences/zoology/geurten_lab/deer_2024/files_extracted/data_with_tortuosity/"
        }
        # You can add more path sets as needed
    }

    path_sys = 'cluster_paths_2'

    rawdata_folder     = paths[path_sys]['rawdata_folder']
    deer_code_filepath = paths[path_sys]['deer_code_filepath']
    merged_signal_file = paths[path_sys]['merged_signal_file']
    plot_file          = paths[path_sys]['plot_file']
    log_file           = paths[path_sys]['log_file']

    saved_loc_and_tort_filepath = paths[path_sys]['saved_loc_and_tort_filepath']

    
    combined_data_filepath=os.path.join(merged_signal_file,deer_code+"Combined.h5")
    combined_data_filepath=os.path.join(saved_loc_and_tort_filepath,deer_code+"_with_tort_and_abs_speed.h5")
    combined_data=load_and_clean_data(combined_data_filepath)
    
    if combined_data.empty:
        raise ValueError("Could not find data for cleaning!")
        
    calculated_tort_speed= pd.DataFrame([])
    calculated_tort_speed=extract_tortuosity.calculate_tortuosity_and_speed(combined_data)
    
    calculated_tort_speed.to_hdf(saved_loc_and_tort_filepath, key='df', mode='w')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Speed and Tortuosity")
    parser.add_argument("deer_code", type=str, help="Deer code to process")
    parser.add_argument("path_sys", type=str, help="Path system to use (e.g., 'alex_paths', 'bart_paths')")
    
    args = parser.parse_args()

    process_data(args.deer_code, args.path_sys)
    
# python your_script_name.py DEERCODE PATHSYSTEM