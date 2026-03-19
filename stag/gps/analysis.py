"""
GPS trajectory processing and feature extraction.

Converts GPS latitude/longitude to New Zealand Map Grid (NZMG)
Cartesian coordinates, computes ground speed and path tortuosity,
applies Gaussian smoothing, and inserts the results into the STAG
database.
"""

import sys
import pandas as pd
import numpy as np
from pyproj import Transformer
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from DeerInfo import DeerDatabaseHandler


def project_to_NZ_map_grid(lats, lons):
    """
    Projects latitude and longitude coordinates to the New Zealand Map Grid (NZMG) system.

    Parameters
    ----------
    lats : Iterable[float]
        An iterable of latitude values.
    lons : Iterable[float]
        An iterable of longitude values.

    Returns
    -------
    tuple
        Two numpy arrays containing the x and y coordinates in the NZMG system.
        """
        # ("EPSG:4326") : LatLon with WGS84 datum used by GPS units and Google Earth
        # ("EPSG:27200") : New Zealad Map Grid of the E_uropean P_etroleum S_urvey G_roup
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:27200")
        xx, yy = transformer.transform(lats, lons)
        return xx,yy

def calculate_tortuosity_and_speed(pos_x, pos_y, fps=50):
    """
    Calculates tortuosity and absolute speed from positional data.

    Parameters
    ----------
    pos_x : np.array
        Array of x coordinates in meters.
    pos_y : np.array
        Array of y coordinates in meters.
    fps : int, optional
        Frames per second of the data collection. Defaults to 50.

    Returns
    -------
    dict
        A dictionary with two keys, 'tortuosity' and 'speed', each associated with a list of calculated values.
        """

    tortuosity_values = []
    absolute_speeds = [] 
    for i in range(len(pos_x) - 2):

        # Vector norm from the Point 0 to Point 1
        vn1 = np.linalg.norm(np.array([pos_x[i+1],pos_y[i+1]])-np.array([pos_x[i],pos_y[i]]))
        
        # Vector norm from the Point 1 to Point 2
        vn2 = np.linalg.norm(np.array([pos_x[i+2],pos_y[i+2]])-np.array([pos_x[i+1],pos_y[i+1]]))
        
        # Vector norm from the Point 0 to Point 2
        vn = np.linalg.norm(np.array([pos_x[i+2],pos_y[i+2]])-np.array([pos_x[i],pos_y[i]]))
        
        #Save to speeds
        absolute_speeds.append(vn1*fps)
        
        # Handle division by zero and save tortuosity
        if vn2 + vn1 == 0:
            tortuosity_values.append(0)  # or any other default value
        else:
            tortuosity_values.append(vn / (vn2 + vn1))
    
    #get last absolute speed        
    i += 1
    # Vector norm from the Poitn 0 to Point 1
    vn1 = np.linalg.norm(np.array([pos_x[i+1],pos_y[i+1]])-np.array([pos_x[i],pos_y[i]]))
    absolute_speeds.append(vn1*fps)
        
    tortuosity_values.insert(0, tortuosity_values[0]) # append 2 zeros for the end 
    tortuosity_values.append(tortuosity_values[-1])
    absolute_speeds.insert(0, absolute_speeds[0]) 
    return {"tortuosity":tortuosity_values,"speed": absolute_speeds}

def update_df_to_cartesian_positions(df):
    """
    Updates a DataFrame with Cartesian coordinates based on latitude and longitude.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'location-lat' and 'location-lon' columns.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with 'pos_x_meter' and 'pos_y_meter' columns added.
        """

    #make relevant subset for transformation
    pos_data = df.loc[:,['location-lat','location-lon']].copy()
    pos_data = pos_data.dropna()
    # get cartesian coordinates
    pos_x_meter,pos_y_meter = project_to_NZ_map_grid(pos_data['location-lat'].to_numpy(),
                                                     pos_data['location-lon'].to_numpy())
    #update subset
    pos_data['pos_x_meter'] = pos_x_meter
    pos_data["pos_y_meter"] = pos_y_meter

    # update original df
    df['pos_x_meter'] = np.nan
    df["pos_y_meter"] = np.nan
    df.update(pos_data)
    return df

def fill_linearly_df(df):
    """
    Interpolates missing values linearly and fills forward and backward in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to fill.

    Returns
    -------
    pd.DataFrame
        The DataFrame with missing values filled linearly, forward, and backward.
        """

    #interpolate missing
    df.interpolate(method='linear', inplace=True)
    # Forward fill 
    df.ffill(inplace=True)
    #backward fill
    df.bfill(inplace=True)

def gaussian_filter_column(df, columnstr, sigma=75):
    """
    Applies a Gaussian filter to a specified column in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column to be filtered.
    columnstr : str
        The name of the column to filter.
    sigma : int, optional
        The sigma parameter of the Gaussian filter. Defaults to 75.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the specified column filtered.
        """


    # Apply the Gaussian filter with a sigma corresponding to a window width of 5
    filtered_signal = gaussian_filter1d(df[columnstr], sigma=sigma)

    df[f'{columnstr}_filt'] = filtered_signal

    return df


def main(filelocation):
    """
    Processes trajectory data from an h5 file and inserts the data into a database.

    The process includes transforming geographic coordinates to Cartesian coordinates, 
    interpolating missing values, applying a Gaussian filter, calculating tortuosity and 
    speed, and inserting the processed data into a specified database.

    Parameters
    ----------
    filelocation : str
        The file path of the h5 file containing trajectory data.

    Returns
    -------
    pd.DataFrame
        The DataFrame with processed data ready for database insertion.
        """

    # load the file
    deer_df = deer_data=pd.read_hdf(filelocation)
    # transform into cartesian coordinates
    deer_df = update_df_to_cartesian_positions(deer_df)
    # fill nan values with interpolation or repetition
    fill_linearly_df(deer_df)
    # low pass filter 
    deer_df = gaussian_filter_column(deer_df,'pos_x_meter')
    deer_df = gaussian_filter_column(deer_df,'pos_y_meter')
    # calculate tortuosity and speed
    result = calculate_tortuosity_and_speed(deer_df.pos_x_meter_filt.to_numpy(),deer_df.pos_y_meter_filt.to_numpy())
    deer_df['abs_speed_mPs'] = result['speed']
    deer_df['tortuosity'] = result['tortuosity']
    # include into the database
    return deer_df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <database_file_position> <h5_file_position>")
        sys.exit(1)

    database_file_position = sys.argv[1]
    h5_file_position = sys.argv[2]

    # Initialize the database handler with the command-line provided database file position
    deer_handler = DeerDatabaseHandler(f'sqlite:///{database_file_position}')
    deer_handler.create_database()

    # Process the h5 file provided via command line
    df = main(h5_file_position)

    # Assuming `insert_trajectory_data_from_h5` is correctly implemented to handle DataFrame `df`
    deer_handler.insert_trajectory_data_from_h5(h5_file_position, df)

