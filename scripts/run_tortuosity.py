import argparse
import os

import extract_tortuosity
import pandas as pd


def load_and_clean_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Data file {filepath} does not exist.")
        return pd.DataFrame([])
    data=pd.read_hdf(filepath)
    return data


def process_data(deer_code, path_sys):

    # Previously a multi-machine path-dict (alex_paths / bart_paths /
    # cluster_paths / cluster_paths_2) lived here.  All paths now come
    # from local_paths.json (per-machine) via stag.local_paths.
    from stag.local_paths import get_path

    # Only the merged-signal file and the tortuosity output dir are
    # actually read in the body below; the previous version of this
    # script defined a multi-machine path dict with every Aoraki
    # directory but most of them were unused.
    merged_signal_file          = get_path("aoraki_merged_signals_v2")
    saved_loc_and_tort_filepath = get_path("aoraki_loc_and_tort_dir")


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
