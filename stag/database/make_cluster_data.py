"""
Extract clustering-ready feature matrices from the STAG database.

Queries the SQLAlchemy database for synchronised accelerometer data
and exports it as a ``.npy`` array suitable for the k-means stage.
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from tqdm import tqdm
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

class DeerInfo(Base):
    """
    Represents information about each deer, including identification and related data.

    Attributes:
        deer_id (Integer): The primary key, autoincrementing.
        repetition_number (String): Identifies the repetition sequence of the data collection for this deer.
        deer_number (String): A unique identifier for the deer.
        accelerometer_data (relationship): Links to associated AccelerometerData records.
        trajectory_data (relationship): Links to associated TrajectoryData records.
    """

    __tablename__ = 'deer_info'
    
    deer_id = Column(Integer, primary_key=True, autoincrement=True)
    repetition_number = Column(String(255), nullable=False)
    deer_number = Column(String(255), nullable=False)
    accelerometer_data = relationship("AccelerometerData", back_populates="deer_info")
    trajectory_data = relationship("TrajectoryData", back_populates="deer_info")

    __table_args__ = ({"mysql_charset": "utf8mb4"},)

class AccelerometerData(Base):
    """
    Stores accelerometer data related to deer movement.

    Attributes:
        data_id (Integer): The primary key, autoincrementing.
        deer_id (Integer): Foreign key linking back to the DeerInfo.
        X_head, Y_head, Z_head (Float): Accelerometer readings for the head.
        X_ear, Y_ear, Z_ear (Float): Accelerometer readings for the ear.
        deer_info (relationship): Back-reference to the associated DeerInfo.
    """

    __tablename__ = 'accelerometer_data'
    
    data_id = Column(Integer, primary_key=True, autoincrement=True)
    deer_id = Column(Integer, ForeignKey('deer_info.deer_id'))
    X_head = Column(Float)
    Y_head = Column(Float)
    Z_head = Column(Float)
    X_ear = Column(Float)
    Y_ear = Column(Float)
    Z_ear = Column(Float)
    deer_info = relationship("DeerInfo", back_populates="accelerometer_data")

    __table_args__ = ({"mysql_charset": "utf8mb4"},)

class TrajectoryData(Base):
    """
    Contains trajectory data including positional information and calculated features.

    Attributes:
        data_id (Integer): The primary key, autoincrementing.
        deer_id (Integer): Foreign key linking back to the DeerInfo.
        pos_WGS84_lat, pos_WGS84_lon (Float): GPS coordinates in the WGS84 system.
        pos_NZMG_x_meter, pos_NZMG_y_meter (Float): Position in the New Zealand Map Grid system.
        pos_x_meter_filt, pos_y_meter_filt (Float): Filtered positional data.
        abs_speed_mPs (Float): Absolute speed in meters per second.
        tortuosity (Float): Calculated tortuosity of the movement path.
        deer_info (relationship): Back-reference to the associated DeerInfo.
    """

    __tablename__ = 'trajectory_data'
    
    data_id = Column(Integer, primary_key=True, autoincrement=True)
    deer_id = Column(Integer, ForeignKey('deer_info.deer_id'))
    pos_WGS84_lat = Column(Float)
    pos_WGS84_lon = Column(Float)
    pos_NZMG_x_meter = Column(Float)
    pos_NZMG_y_meter = Column(Float)
    pos_x_meter_filt = Column(Float)
    pos_y_meter_filt = Column(Float)
    abs_speed_mPs = Column(Float)
    tortuosity    = Column(Float)
    deer_info = relationship("DeerInfo", back_populates="trajectory_data")

    __table_args__ = ({"mysql_charset": "utf8mb4"},)

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
    database_url = 'sqlite:////projects/sciences/zoology/geurten_lab/deer_2024/deer_data.db'  # Update this to your database URL
    session, engine = open_session(database_url)
    deer_ids = get_deer_ids(session)
    
    all_data = aggregate_all_data(session, deer_ids)
    save_data_to_npy(all_data, '/projects/sciences/zoology/geurten_lab/deer_2024/clust_data_raw.npy')

    session.close()
    print("Data processing complete and saved to '/projects/sciences/zoology/geurten_lab/deer_2024/clust_data_raw.npy'.")
