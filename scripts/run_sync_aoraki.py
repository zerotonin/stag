import pandas as pd
from Better_Data_Sync import Better_Data_Sync as ds
import os
import argparse

def load_and_clean_data(filepath,ear=False):
    if not os.path.exists(filepath):
        print(f"Error: Data file {filepath} does not exist.")
        return pd.DataFrame([])
    if ear==False:
        columns_to_keep=['TagID', 'Date', 'Time', 'X', 'Y', 'Z', 'location-lat','location-lon']
    else:
        columns_to_keep=['TagID', 'Date', 'Time', 'X', 'Y', 'Z']
    data=pd.read_csv(filepath,usecols=columns_to_keep)
    data['DateTime_Global'] = pd.to_datetime(data['Date'] + ' ' + data['Time'],format=r"%d/%m/%Y %H:%M:%S.%f")
    data['NZ_DateTime'] = data['DateTime_Global'] + pd.Timedelta(hours=13)
    data.drop(['Date', 'Time'], axis=1, inplace=True)
    return data


def process_data(deer_code, path_sys):

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


    rawdata_folder     = paths[path_sys]['rawdata_folder']
    deer_code_filepath = paths[path_sys]['deer_code_filepath']
    merged_signal_file = paths[path_sys]['merged_signal_file']
    plot_file          = paths[path_sys]['plot_file']
    log_file           = paths[path_sys]['log_file']

    
    ear_data_filepath=os.path.join(rawdata_folder,deer_code+"_LE.csv")
    head_data_filepath=os.path.join(rawdata_folder,deer_code+"_LH.csv")

        
    ear_data=load_and_clean_data(ear_data_filepath,ear=True)
    head_data=load_and_clean_data(head_data_filepath,ear=False)
        
    if head_data.empty or ear_data.empty:
        raise ValueError("Could not find data for cleaning!")
        
    window_dict = {
    'end_window_beginning_index_head': head_data.shape[0] - 20000,
    'end_window_end_index_head': head_data.shape[0],
    'end_window_beginning_index_ear': ear_data.shape[0] - 20000,
    'end_window_end_index_ear': ear_data.shape[0],
    'start_window_beginning_index_head': 0,
    'start_window_end_index_head': 20000,
    'start_window_beginning_index_ear': 0,
    'start_window_end_index_ear': 20000
    }      
        

        
    syncer= ds(deer_id=deer_code,head_data=head_data,ear_data=ear_data,window_dict=window_dict,log=True,log_folder=log_file,mkplot=True,plot_folder=plot_file)
    syncer.find_signal_match()
    syncer.interpolate_and_save_signal(merged_signal_file)

    syncer.close_logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process deer data.")
    parser.add_argument("deer_code", type=str, help="Deer code to process")
    parser.add_argument("path_sys", type=str, help="Path system to use (e.g., 'alex_paths', 'bart_paths')")
    
    args = parser.parse_args()

    process_data(args.deer_code, args.path_sys)
    
# python your_script_name.py DEERCODE PATHSYSTEM