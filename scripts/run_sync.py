import os

import pandas as pd
from Better_Data_Sync import Better_Data_Sync as ds

from stag.local_paths import get_path


def load_and_clean_data(filepath, ear=False):
    if not os.path.exists(filepath):
        print(f"Error: Data file {filepath} does not exist.")
        return pd.DataFrame([])
    if not ear:
        columns_to_keep = [
            "TagID", "Date", "Time", "X", "Y", "Z",
            "location-lat", "location-lon",
        ]
    else:
        columns_to_keep=['TagID', 'Date', 'Time', 'X', 'Y', 'Z']
    data=pd.read_csv(filepath,usecols=columns_to_keep)
    data['DateTime_Global'] = pd.to_datetime(data['Date'] + ' ' + data['Time'],format=r"%d/%m/%Y %H:%M:%S.%f")
    data['NZ_DateTime'] = data['DateTime_Global'] + pd.Timedelta(hours=13)
    data.drop(['Date', 'Time'], axis=1, inplace=True)
    return data


# Previously a multi-machine path-dict (alex_paths / bart_paths /
# cluster_paths) lived here.  All of it is now in local_paths.json.
rawdata_folder     = get_path("aoraki_raw_data")
deer_code_filepath = get_path("aoraki_deer_codes")
merged_signal_file = get_path("aoraki_merged_signals")
plot_file          = get_path("aoraki_plot_dir")
log_file           = get_path("aoraki_log_dir")




with open(deer_code_filepath, 'r') as file:
    # Read the lines
    deer_lines = file.readlines()
deer_codes=[line.strip() for line in deer_lines]



first_iteration=True
for deer_code in deer_codes:

    ear_data_filepath=os.path.join(rawdata_folder,deer_code+"_LE.csv")
    head_data_filepath=os.path.join(rawdata_folder,deer_code+"_LH.csv")


    ear_data=load_and_clean_data(ear_data_filepath,ear=True)
    head_data=load_and_clean_data(head_data_filepath,ear=False)

    if head_data.empty or ear_data.empty:
        continue # keep going if doesnt exist

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




    if first_iteration:
        syncer = ds(
            deer_id=deer_code, head_data=head_data, ear_data=ear_data,
            window_dict=window_dict,
            log=True, log_folder=log_file,
            mkplot=True, plot_folder=plot_file,
        )
        syncer.find_signal_match()
        syncer.interpolate_and_save_signal(merged_signal_file)
        first_iteration=False
    else:
        syncer.change_data(deer_code,head_data,ear_data,window_dict)
        syncer.find_signal_match()
        syncer.interpolate_and_save_signal(merged_signal_file)


syncer.close_logger()
