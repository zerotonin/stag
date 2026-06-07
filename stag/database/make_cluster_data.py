# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — database.make_cluster_data                               ║
# ║  « DB → clustering-ready .npy feature matrix »                   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Queries the SQLAlchemy database for synchronised accelerometer ║
# ║  and trajectory data per deer and concatenates everything into  ║
# ║  a single .npy array suitable for the k-means stage.            ║
# ║                                                                  ║
# ║  ORM models are imported from stag.database.orm — do not        ║
# ║  redeclare them here.                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""DB → clustering-ready .npy feature matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from stag.database.orm import AccelerometerData, DeerInfo, TrajectoryData


def open_session(database_url):
    """Opens a session for the database."""
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    return Session(), engine

def get_deer_ids(session):
    """Returns a list of all deer_ids in the database."""
    deer_ids = session.query(DeerInfo.deer_id).all()
    return [deer_id[0] for deer_id in deer_ids]

def get_data_for_deer(session, deer_id):
    """Fetches and concatenates accelerometer and trajectory data for a given deer_id."""
    acc_data = pd.read_sql(session.query(AccelerometerData).filter_by(deer_id=deer_id).statement, session.bind)
    traj_data = pd.read_sql(session.query(TrajectoryData).filter_by(deer_id=deer_id).statement, session.bind)

    # Selecting relevant columns and interpolating NaN values
    acc_data = acc_data[['X_head', 'Y_head', 'Z_head', 'X_ear', 'Y_ear', 'Z_ear']].interpolate()
    traj_data = traj_data[['abs_speed_mPs', 'tortuosity']].interpolate()

    # Concatenating horizontally
    combined_data = pd.concat([acc_data, traj_data], axis=1).dropna()
    return combined_data.values

def aggregate_all_data(session, deer_ids):
    """Aggregates data for all deer_ids and interpolates over NaN values."""
    all_data = []
    for deer_id in tqdm(deer_ids, desc="Processing Deer IDs"):
        deer_data = get_data_for_deer(session, deer_id)
        all_data.append(deer_data)

    # Stack vertically
    return np.vstack(all_data)

def save_data_to_npy(data, filename):
    """Saves the data to a .npy file."""
    np.save(filename, np.array(data, dtype=float))

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from stag.local_paths import get_path

    parser = argparse.ArgumentParser(
        description="Aggregate per-deer accelerometer data from the legacy "
                    "SQLite database into a single clust_data_raw.npy.  "
                    "DB URL defaults to local_paths.json deer_db_url_legacy.",
    )
    parser.add_argument(
        "--db-url", default=None,
        help="SQLAlchemy URL for the source database (default: resolve "
             "STAG_DEER_DB_URL_LEGACY or local_paths.json deer_db_url_legacy).",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Output .npy path (this script previously hard-coded an "
             "Aoraki path; it is now a required argument).",
    )
    args = parser.parse_args()

    database_url = args.db_url or get_path("deer_db_url_legacy")
    session, engine = open_session(database_url)
    deer_ids = get_deer_ids(session)

    all_data = aggregate_all_data(session, deer_ids)
    save_data_to_npy(all_data, args.output)

    session.close()
    print(f"Data processing complete and saved to '{args.output}'.")
