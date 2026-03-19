"""
Tortuosity and speed from raw GPS latitude/longitude.

Haversine-based distance calculation between consecutive GPS fixes,
yielding arc-chord tortuosity ratios and absolute ground speed.
"""
import pandas as pd
import math
import numpy as np



def calculate_tortuosity_and_speed(lat,lon,fps=0.5):
    tortuosity_values = []
    absolute_speeds = [] 
    for i in range(len(lat) - 2):
        # Vector norm from the Point 0 to Point 1
        vn1 = lat_lon_vec_to_meter_vec(lat[i],lon[i],lat[i+1],lon[i+1])
        
        # Vector norm from the Point 1 to Point 2
        vn2 = lat_lon_vec_to_meter_vec(lat[i+1],lon[i+1],lat[i+2],lon[i+2])
        
        # Vector norm from the Point 0 to Point 2
        vn = lat_lon_vec_to_meter_vec(lat[i],lon[i],lat[i+2],lon[i+2])
        
        absolute_speeds.append(vn1/fps)
        
        # Handle division by zero
        if vn2 + vn1 == 0:
            tortuosity_values.append(0)  # or any other default value
        else:
            tortuosity_values.append(vn / (vn2 + vn1))
    
    #get last absolute speed        
    i += 1
    # Vector norm from the Poitn 0 to Point 1
    vn1 = lat_lon_vec_to_meter_vec(lat[i],lon[i],lat[i+1],lon[i+1])
    absolute_speeds.append(vn1/fps)
        
    tortuosity_values.insert(0, tortuosity_values[0]) # append 2 zeros for the end 
    tortuosity_values.append(tortuosity_values[-1])
    absolute_speeds.insert(0, absolute_speeds[0]) 
    return {"tortuosity":tortuosity_values,"speed": absolute_speeds}

def lat_lon_vec_to_meter_vec(lat1, lon1, lat2, lon2):  # generally used geo measurement function
    R = 6378.137 # Radius of earth in KM
    dLat = lat2 * np.pi / 180 - lat1 * np.pi / 180
    dLon = lon2 * np.pi / 180 - lon1 * np.pi / 180
    
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d * 1000  #meters

def extract_tort_and_speed(saved_loc_and_tort_filepath):
    deer_data=pd.read_hdf(saved_loc_and_tort_filepath)
    deer_data.head()

    pos_data = deer_data.loc[:,['location-lat','location-lon']].copy()
    pos_data = pos_data.dropna()
    pos_data.head()
    result_dict = calculate_tortuosity_and_speed(pos_data['location-lat'].to_numpy(),
                                                pos_data['location-lon'].to_numpy())
    print(result_dict)

    pos_data['tortuosity']= result_dict['tortuosity']
    pos_data["absolute_speed"]= result_dict["speed"]
    pos_data.head()

    original_deer_data=deer_data.loc[:,['location-lat','location-lon']].copy()
    original_deer_data['tortuosity'] = pd.NA
    original_deer_data["absolute_speed"] = pd.NA
    original_deer_data.update(pos_data)

    deer_data_interpolated=original_deer_data.copy()
    deer_data_interpolated['location-lat'].interpolate(method='linear', inplace=True) # linearly interpolate all Nas between obs of location
    deer_data_interpolated['location-lon'].interpolate(method='linear', inplace=True)
    deer_data_interpolated['tortuosity'].interpolate(method='linear', inplace=True)
    deer_data_interpolated['absolute_speed'].interpolate(method='linear', inplace=True)


    deer_data_interpolated['location-lat'].bfill(inplace=True) # bfill and FFill any extra Nas
    deer_data_interpolated['location-lon'].bfill(inplace=True)
    deer_data_interpolated['location-lat'].ffill(inplace=True)
    deer_data_interpolated['location-lon'].ffill(inplace=True)
    deer_data_interpolated['tortuosity'].ffill(inplace=True)
    deer_data_interpolated['tortuosity'].bfill(inplace=True)
    deer_data_interpolated['absolute_speed'].ffill(inplace=True)
    deer_data_interpolated['absolute_speed'].bfill(inplace=True)
    return deer_data_interpolated


