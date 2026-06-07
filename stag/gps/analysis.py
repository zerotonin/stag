# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — gps.analysis                                             ║
# ║  « ground speed and NZMG positions »                             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Project WGS84 GPS samples to the New Zealand Map Grid           ║
# ║  (NZMG, EPSG:27200), compute ground speed and path               ║
# ║  tortuosity, and apply Gaussian smoothing.                       ║
# ║                                                                  ║
# ║  Indentation rot in several function bodies (pre-existing) is    ║
# ║  fixed in this revision; the broken ``from DeerInfo import ...`` ║
# ║  reference left over from the headshake_project rename now       ║
# ║  points at the canonical ``stag.database.handler`` module.       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""GPS trajectory processing and feature extraction.

Project WGS84 GPS samples to NZMG, fill linear gaps, apply a Gaussian
filter to the projected positions, and compute per-sample tortuosity
and ground speed.  Driver block (:func:`main`) reads a single h5
trajectory file, processes it, and inserts the result into a deer-data
SQLite database via :class:`stag.database.handler.DeerDatabaseHandler`.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def project_to_NZ_map_grid(lats, lons):
    """Project (lat, lon) WGS84 coordinates to the New Zealand Map Grid.

    Args:
        lats: Iterable of latitude values (WGS84, EPSG:4326).
        lons: Iterable of longitude values (WGS84, EPSG:4326).

    Returns:
        Tuple ``(xx, yy)`` of NZMG x and y coordinates (EPSG:27200).
    """
    # pyproj is only needed when this function is actually called;
    # deferred so `import stag.gps.analysis` works on any machine.
    from pyproj import Transformer

    # EPSG:4326 — WGS84 lat/lon used by GPS units and Google Earth.
    # EPSG:27200 — New Zealand Map Grid (European Petroleum Survey Group).
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27200")
    xx, yy = transformer.transform(lats, lons)
    return xx, yy


def calculate_tortuosity_and_speed(pos_x, pos_y, fps=50):
    """Compute tortuosity and absolute speed from a trajectory.

    Args:
        pos_x: Array of x coordinates (metres).
        pos_y: Array of y coordinates (metres).
        fps:   Sample rate in Hz (default 50).

    Returns:
        Dict with ``"tortuosity"`` and ``"speed"`` lists, one entry
        per sample (boundary samples duplicated to match input length).
    """
    tortuosity_values = []
    absolute_speeds = []
    for i in range(len(pos_x) - 2):
        # |v0->1|
        vn1 = np.linalg.norm(
            np.array([pos_x[i + 1], pos_y[i + 1]])
            - np.array([pos_x[i], pos_y[i]]),
        )
        # |v1->2|
        vn2 = np.linalg.norm(
            np.array([pos_x[i + 2], pos_y[i + 2]])
            - np.array([pos_x[i + 1], pos_y[i + 1]]),
        )
        # |v0->2|
        vn = np.linalg.norm(
            np.array([pos_x[i + 2], pos_y[i + 2]])
            - np.array([pos_x[i], pos_y[i]]),
        )

        absolute_speeds.append(vn1 * fps)

        if vn2 + vn1 == 0:
            tortuosity_values.append(0)
        else:
            tortuosity_values.append(vn / (vn2 + vn1))

    # Trailing speed for the last sample.
    i += 1
    vn1 = np.linalg.norm(
        np.array([pos_x[i + 1], pos_y[i + 1]])
        - np.array([pos_x[i], pos_y[i]]),
    )
    absolute_speeds.append(vn1 * fps)

    # Pad both ends so output length matches input length.
    tortuosity_values.insert(0, tortuosity_values[0])
    tortuosity_values.append(tortuosity_values[-1])
    absolute_speeds.insert(0, absolute_speeds[0])
    return {"tortuosity": tortuosity_values, "speed": absolute_speeds}


def update_df_to_cartesian_positions(df):
    """Add NZMG ``pos_x_meter`` and ``pos_y_meter`` columns to ``df``.

    Args:
        df: DataFrame with ``location-lat`` and ``location-lon`` columns.

    Returns:
        The same DataFrame with the two new metre columns populated.
    """
    pos_data = df.loc[:, ["location-lat", "location-lon"]].copy()
    pos_data = pos_data.dropna()
    pos_x_meter, pos_y_meter = project_to_NZ_map_grid(
        pos_data["location-lat"].to_numpy(),
        pos_data["location-lon"].to_numpy(),
    )
    pos_data["pos_x_meter"] = pos_x_meter
    pos_data["pos_y_meter"] = pos_y_meter

    df["pos_x_meter"] = np.nan
    df["pos_y_meter"] = np.nan
    df.update(pos_data)
    return df


def fill_linearly_df(df):
    """Linearly interpolate then forward / backward-fill ``df``.

    Args:
        df: DataFrame to fill in place.
    """
    df.interpolate(method="linear", inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)


def gaussian_filter_column(df, columnstr, sigma=75):
    """Apply a Gaussian filter to one column of a DataFrame.

    Args:
        df:        DataFrame with the source column.
        columnstr: Name of the column to filter.
        sigma:     Gaussian sigma (default 75).

    Returns:
        ``df`` with a new ``{columnstr}_filt`` column appended.
    """
    filtered_signal = gaussian_filter1d(df[columnstr], sigma=sigma)
    df[f"{columnstr}_filt"] = filtered_signal
    return df


def main(filelocation):
    """Process one h5 trajectory file into a DataFrame ready for DB ingest.

    Transforms (lat, lon) to NZMG metres, interpolates NaN gaps,
    Gaussian-smooths the positions, and computes per-sample tortuosity
    and ground speed.

    Args:
        filelocation: Path to the h5 trajectory file.

    Returns:
        DataFrame with the filtered positions, speed and tortuosity
        columns, ready for :meth:`DeerDatabaseHandler.
        insert_trajectory_data_from_h5`.
    """
    deer_df = pd.read_hdf(filelocation)
    deer_df = update_df_to_cartesian_positions(deer_df)
    fill_linearly_df(deer_df)
    deer_df = gaussian_filter_column(deer_df, "pos_x_meter")
    deer_df = gaussian_filter_column(deer_df, "pos_y_meter")
    result = calculate_tortuosity_and_speed(
        deer_df.pos_x_meter_filt.to_numpy(),
        deer_df.pos_y_meter_filt.to_numpy(),
    )
    deer_df["abs_speed_mPs"] = result["speed"]
    deer_df["tortuosity"] = result["tortuosity"]
    return deer_df


if __name__ == "__main__":
    from stag.database.handler import DeerDatabaseHandler

    if len(sys.argv) != 3:
        print(
            "Usage: python -m stag.gps.analysis "
            "<database_file_position> <h5_file_position>",
        )
        sys.exit(1)

    database_file_position = sys.argv[1]
    h5_file_position = sys.argv[2]

    deer_handler = DeerDatabaseHandler(f"sqlite:///{database_file_position}")
    deer_handler.create_database()

    df = main(h5_file_position)
    deer_handler.insert_trajectory_data_from_h5(h5_file_position, df)
