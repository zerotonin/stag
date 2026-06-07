import argparse
import os

import pandas as pd
from Better_Data_Sync import Better_Data_Sync as ds


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


def process_data(deer_code, path_sys):
    """Run the sync pipeline for one deer.

    ``path_sys`` selects which Aoraki signal directory to write to:
      - ``"cluster_paths"``    -> the legacy sync_file_results/ tree
      - ``"cluster_paths_2"``  -> the active new_file_sync/ tree

    Both sets of paths are resolved by :mod:`stag.local_paths`; the
    previous multi-machine inline dict (alex_paths / bart_paths /
    cluster_paths / cluster_paths_2 with absolute Windows + Linux
    paths) is gone.
    """
    from stag.local_paths import get_path

    if path_sys == "cluster_paths":
        merged_key = "aoraki_merged_signals"
        plot_key   = "aoraki_plot_dir"
        log_key    = "aoraki_log_dir"
    elif path_sys == "cluster_paths_2":
        merged_key = "aoraki_merged_signals_v2"
        plot_key   = "aoraki_plot_dir_v2"
        log_key    = "aoraki_log_dir_v2"
    else:
        raise ValueError(
            f"unknown path_sys {path_sys!r}; expected 'cluster_paths' "
            "or 'cluster_paths_2'",
        )

    rawdata_folder     = get_path("aoraki_raw_data")
    merged_signal_file = get_path(merged_key)
    plot_file          = get_path(plot_key)
    log_file           = get_path(log_key)


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



    syncer = ds(
        deer_id=deer_code, head_data=head_data, ear_data=ear_data,
        window_dict=window_dict,
        log=True, log_folder=log_file,
        mkplot=True, plot_folder=plot_file,
    )
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
