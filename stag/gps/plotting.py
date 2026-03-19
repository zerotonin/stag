"""
GPS trajectory visualisation.

Functions for plotting deer trajectories as colour-coded line
collections with optional close-up insets.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def load_trajectory_data(csv_file):
    """
    Load trajectory data from a CSV file.

    Parameters:
    - csv_file: Path to the CSV file containing trajectory data.

    Returns
    -------
        - DataFrame with the trajectory data.
        """
        return pd.read_csv(csv_file)

def prepare_line_collection(df, x_col, y_col, start_idx=None, end_idx=None, cmap='Greys', full_length_color=None):
    """
    Prepare a LineCollection for plotting trajectory data with an option to match color coding with a larger plot.
    
    - full_length_color: Pass the np.linspace color array used for the full plot to ensure continuity in the close-up.
    """
    if start_idx is None or end_idx is None:
        points = np.array([df[x_col], df[y_col]]).T.reshape(-1, 1, 2)
        color_array = np.linspace(0.3, 1, len(points)) if full_length_color is None else full_length_color
    else:
        points = np.array([df[x_col][start_idx:end_idx], df[y_col][start_idx:end_idx]]).T.reshape(-1, 1, 2)
        # Use a subset of the full plot's color array to match the colors exactly
        color_array = full_length_color[start_idx:end_idx] if full_length_color is not None else np.linspace(0.3, 1, len(points))
    
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0.3, 1))
    lc.set_array(color_array)
    lc.set_linewidth(2)
    
    return lc



def plot_trajectory(ax, line_collection, x_label, y_label):
    """
    Plot a trajectory on a given axis.

    Parameters:
    - ax: The axis on which to plot.
    - line_collection: The LineCollection object representing the trajectory.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    """
    ax.add_collection(line_collection)
    ax.autoscale()
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)  # Adding grid

def highlight_area(ax, df, x_col, y_col, start_idx, end_idx, edgecolor='black'):
    """
    Highlight an area on the plot.

    Parameters:
    - ax: The axis on which to plot.
    - df: DataFrame containing trajectory data.
    - x_col: Name of the column for x-axis values.
    - y_col: Name of the column for y-axis values.
    - start_idx: Start index of the area to highlight.
    - end_idx: End index of the area to highlight.
    - edgecolor: Color of the highlighting rectangle.
    """
    min_x, max_x = df.iloc[start_idx:end_idx][x_col].min(), df.iloc[start_idx:end_idx][x_col].max()
    min_y, max_y = df.iloc[start_idx:end_idx][y_col].min(), df.iloc[start_idx:end_idx][y_col].max()
    ax.add_patch(plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor=edgecolor, linewidth=3,zorder =3))



def main_plot_with_closeup(df, start_idx, end_idx, coord_system, line_color = 'Greys'):


    # Define columns based on the coordinate system
    if coord_system == 'WGS':
        x_col, y_col = 'pos_WGS84_lon', 'pos_WGS84_lat'
        x_label, y_label = 'Longitude', 'Latitude'
    elif coord_system == 'NZMG':
        x_col, y_col = 'pos_NZMG_x_meter', 'pos_NZMG_y_meter'
        x_label, y_label = 'Meter X', 'Meter Y'
    elif coord_system == 'FILT':
        x_col, y_col = 'pos_x_meter_filt', 'pos_y_meter_filt'
        x_label, y_label = 'Filtered Meter X', 'Filtered Meter Y'
    else:
        raise ValueError("Invalid coordinate system specified. Use 'WGS', 'NZMG', or 'FILT'.")
    

    # Normalize coordinates if not WGS
    if coord_system != "WGS":
        df[x_col] -= df[x_col].min()
        df[y_col] -= df[y_col].min()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})
    
    # Color array for the full plot
    full_length_color = np.linspace(0.3, 1, len(df))

    # Full trajectory with the entire color range
    lc_full = prepare_line_collection(df, x_col, y_col, cmap=line_color, full_length_color=full_length_color)
    plot_trajectory(ax1, lc_full, x_label, y_label)

    # Close-up view using the same color array segment
    lc_zoom = prepare_line_collection(df, x_col, y_col, start_idx, end_idx, cmap=line_color, full_length_color=full_length_color)
    plot_trajectory(ax2, lc_zoom, x_label, y_label)

    highlight_area(ax1, df, x_col, y_col, start_idx, end_idx)

    plt.tight_layout()
    return fig

def calculate_time_axis(df, fps=50):
    """
    Calculate time axis for the DataFrame based on the FPS.

    Parameters:
    - df: DataFrame containing the data.
    - fps: Frames per second (default: 50).

    Returns
    -------
        - A NumPy array representing the time in seconds.
        """
        num_points = len(df)
        return np.linspace(0, num_points / fps, num_points)

def plot_speed_and_tortuosity_with_highlight(df, start_idx, end_idx, fps=50):
    time_seconds = calculate_time_axis(df,fps)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Absolute Speed on the first subplot
    color_speed = 'tab:red'
    ax1.plot(time_seconds, df['abs_speed_mPs'], color=color_speed, label='Abs Speed')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Abs Speed (m/s)', color=color_speed)
    ax1.tick_params(axis='y', labelcolor=color_speed)
    
    # Create a twin Axes for Tortuosity on the first subplot
    ax1_twin = ax1.twinx()
    color_tort = 'tab:blue'
    ax1_twin.plot(time_seconds, df['tortuosity'], color=color_tort, label='Tortuosity')
    ax1_twin.set_ylabel('Tortuosity', color=color_tort)
    ax1_twin.tick_params(axis='y', labelcolor=color_tort)
    
    # Plot the same data on the second subplot for the close-up view
    ax2.plot(time_seconds[start_idx:end_idx], df['abs_speed_mPs'][start_idx:end_idx], color=color_speed)
    ax2.plot(time_seconds[start_idx:end_idx], df['tortuosity'][start_idx:end_idx], color=color_tort)
    
    # Adjust the second subplot to focus on the specified segment
    ax2.set_xlim(time_seconds[start_idx], time_seconds[end_idx])
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Abs Speed (m/s) / Tortuosity')
    
    # Optional: Synchronize y-axis limits for both subplots if desired
    ax2.set_ylim(ax1.get_ylim())
    ax2.tick_params(axis='y', labelcolor=color_speed)
    
    fig.tight_layout()
    return fig


# Replace the path with your actual CSV file path
csv_file =  '/home/geuba03p/deer_accl/deer_2_trajectory.csv'  # Replace with the path to your CSV file
df = load_trajectory_data(csv_file)
fig_list = list()
filename_list = ['WGS',"NZMG","NZMGF",'speed']
extension_list = ['png','png','png','svg']
fig_list.append(main_plot_with_closeup(df, start_idx=4000000, end_idx=4010000, coord_system='WGS',line_color='Greens'))
fig_list.append(main_plot_with_closeup(df, start_idx=4000000, end_idx=4010000, coord_system='NZMG',line_color='Blues'))
fig_list.append(main_plot_with_closeup(df, start_idx=4000000, end_idx=4010000, coord_system='FILT',line_color='Oranges'))
fig_list.append(plot_speed_and_tortuosity_with_highlight(df, 4000000, 4010000))

for fig, fname in list(zip(fig_list,filename_list)):
    fig.savefig(f"/home/geuba03p/deer_accl/deer_2_{fname}.png")
