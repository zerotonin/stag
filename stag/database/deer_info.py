"""
SQLAlchemy ORM models and database handler for deer sensor data.

Defines the database schema (DeerInfo, AccelerometerData,
TrajectoryData, VideoAvailability) and a handler class for creating
the database, ingesting HDF5 files, and querying deer records.
"""

import os,sys,csv,json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
import datetime
from tqdm import tqdm
from scipy import stats

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
    NZ_DateTime = Column(DateTime)
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

class VideoObservationReference(Base):
    """
    Represents a reference to a video observation related to deer behavior analysis.

    Attributes:
        id (Integer): Primary key, unique identifier for each video observation.
        original_file_path (String): The file path of the video observation.
        repetition (Integer): An identifier for the repetition sequence of the data collection for this video.
        deer (Integer): A unique identifier for the deer observed in the video.
        frame_count (Integer): The total number of frames in the video observation.
        start_time (DateTime): The starting time of the video observation.
        stop_time (DateTime): The ending time of the video observation, calculated based on the frame count and frames per second (FPS).
        comment (String): Optional field for additional comments or notes about the video observation.

    Relationships:
        video_availability: A list of VideoAvailability instances that associate this video observation with corresponding accelerometer data points.
    """
    __tablename__ = 'video_observation_reference'
    id = Column(Integer, primary_key=True)
    original_file_path = Column(String)
    repetition = Column(Integer)
    deer = Column(Integer)
    frame_count = Column(Integer)
    start_time = Column(DateTime)
    stop_time = Column(DateTime)
    comment = Column(String)
    __table_args__ = ({"mysql_charset": "utf8mb4"},)

class VideoAvailability(Base):
    """
    Links accelerometer data points with video observations, indicating frame-specific associations and availability.

    Attributes:
        id (Integer): Primary key, unique identifier for each video availability record.
        accelerometer_data_id (Integer): Foreign key linking to an AccelerometerData record.
        video_observation_reference_id (Integer, nullable): Foreign key linking to a VideoObservationReference record. Nullable to indicate cases where accelerometer data does not have an associated video observation.
        frame (Integer, nullable): Specifies the frame number in the video that corresponds to the accelerometer data point. Nullable to indicate that specific frame information is not available or applicable.

    Relationships:
        accelerometer_data: The AccelerometerData instance this availability record is associated with.
        video_observation_reference: The VideoObservationReference instance this availability record is associated with. Nullable if there is no direct video observation association.

    Note:
        The absence (NULL value) of video_observation_reference_id or frame implicitly indicates that the video is not available or a specific frame is not designated for the corresponding accelerometer data point.
    """
    __tablename__ = 'video_availability'
    id = Column(Integer, primary_key=True)
    accelerometer_data_id = Column(Integer, ForeignKey('accelerometer_data.data_id'))
    video_observation_reference_id = Column(Integer, ForeignKey('video_observation_reference.id'), nullable=True)
    frame = Column(Integer, nullable=True)

    accelerometer_data = relationship("AccelerometerData", backref="video_availability")
    video_observation_reference = relationship("VideoObservationReference", backref="video_availability")

    __table_args__ = ({"mysql_charset": "utf8mb4"},)

class ClusterLabels(Base):
    __tablename__ = 'cluster_labels'
    data_id = Column(Integer, ForeignKey('accelerometer_data.data_id'), primary_key=True)
    label = Column(Integer, nullable=False)

    accelerometer_data = relationship("AccelerometerData", backref="cluster_labels")
    __table_args__ = ({"mysql_charset": "utf8mb4"},)



class DeerDatabaseHandler:
    """Handles interactions with the deer behavior database.

    This class provides methods to create the database schema, insert data from h5 files, 
    and query the database for deer ids based on filenames. It is designed to work with 
    a specific schema consisting of DeerInfo, AccelerometerData, and TrajectoryData tables.

    Attributes:
        engine (sqlalchemy.engine.Engine): The database engine.
    """
    def __init__(self, database_url):
        """Initializes the DeerDatabaseHandler with a database engine.

        Parameters
        ----------
        database_url : str
            The database connection string.
            """
            self.engine = create_engine(database_url)


    def create_database(self):
        """Creates the database schema based on declarative base classes."""
        Base.metadata.create_all(self.engine)

    @staticmethod
    def read_h5_file(filename):
        """Reads an h5 file and returns its contents as a DataFrame.

        Parameters
        ----------
        filename : str
            The path to the h5 file.

        Returns
        -------
        pd.DataFrame
            The data from the h5 file.
            """
            data = pd.read_hdf(filename, 'df')
            return data

    def insert_data_from_directory(self, directory):
        """Inserts data from h5 files located in a directory into the database.

        Parameters
        ----------
        directory : str
            The path to the directory containing h5 files.
            """
            session = sessionmaker(bind=self.engine)()
    
        for filename in os.listdir(directory):
            if filename.endswith('.h5'):
                repetition, deer = filename.split('_')[1], filename.split('_')[2]
                data = self.read_h5_file(os.path.join(directory, filename))
                
                deer_info = DeerInfo(repetition_number=repetition, deer_number=deer)
                session.add(deer_info)
                session.flush()

                for _, row in data.iterrows():
                    accelerometer_data = AccelerometerData(deer_id=deer_info.deer_id, X_head=row['X_head'], Y_head=row['Y_head'], Z_head=row['Z_head'], X_ear=row['X_ear'], Y_ear=row['Y_ear'], Z_ear=row['Z_ear'])
                    session.add(accelerometer_data)
                
                session.commit()
        session.close()

    def get_deer_id_from_filename(self, filename):
        """Queries the database for a deer_id based on the filename.

        Parameters
        ----------
        filename : str
            The filename containing the repetition and deer numbers.

        Returns
        -------
            int or None: The deer_id if found, or None if not found.
            """
            session = sessionmaker(bind=self.engine)()
            # Assuming filename format: "anything_R{repetition}_D{deer}.h5"
            parts = filename.split('_')
            repetition_part = [part for part in parts if part.startswith('R')]
            deer_part = [part for part in parts if part.startswith('D')]

        if repetition_part and deer_part:
        
            # Query the database
            deer_info = session.query(DeerInfo).filter_by(repetition_number=repetition_part[0], deer_number=deer_part[0]).first()

            if deer_info:
                return deer_info.deer_id

        return None

    def insert_trajectory_data_from_h5(self, file_position, data):
        """Inserts trajectory data from an h5 file into the database.

        Parameters
        ----------
        file_position : str
            The path to the h5 file.
        data : pd.DataFrame
            The DataFrame containing trajectory data.
            """
            session = sessionmaker(bind=self.engine)()

    
        if file_position.endswith('.h5'):
            deer_id = self.get_deer_id_from_filename(os.path.basename(file_position))
            if deer_id:
                for _, row in data.iterrows():
                    trajectory_data = TrajectoryData(deer_id=deer_id, 
                                                    pos_WGS84_lat = row['location-lat'],
                                                    pos_WGS84_lon=row['location-lon'], 
                                                    pos_NZMG_x_meter=row['pos_x_meter'], 
                                                    pos_NZMG_y_meter=row['pos_y_meter'], 
                                                    pos_x_meter_filt=row['pos_x_meter_filt'], 
                                                    pos_y_meter_filt=row['pos_y_meter_filt'],
                                                    abs_speed_mPs=row['abs_speed_mPs'],  
                                                    tortuosity=row['tortuosity'])
                    session.add(trajectory_data)

                session.commit()
            else:
                "This deer is not currently in the database please add the accelometer data first or register it otherwise!"
        session.close()
    
    def write_trajectory_data_for_deer(self, deer_number, output_file=None):
        """
        Queries the trajectory data for a specific deer and writes it to a file or prints it.

        Parameters
        ----------
        deer_number : str
            The unique identifier for the deer.
        output_file : str, optional
            The path to the output file. If None, prints to standard output.
            """
            session = sessionmaker(bind=self.engine)()

        # Fetch the deer_id using deer_number
        deer_info = session.query(DeerInfo).filter_by(deer_id=deer_number).first()
        if not deer_info:
            print(f"No data found for deer_number: {deer_number}")
            return
        
        # Query the trajectory data for the specific deer_id
        trajectory_data = session.query(TrajectoryData).filter_by(deer_id=deer_info.deer_id).all()
        
        # Convert to DataFrame
        data = pd.DataFrame([{
            'data_id': datum.data_id,
            'pos_WGS84_lat': datum.pos_WGS84_lat,
            'pos_WGS84_lon': datum.pos_WGS84_lon,
            'pos_NZMG_x_meter': datum.pos_NZMG_x_meter,
            'pos_NZMG_y_meter': datum.pos_NZMG_y_meter,
            'pos_x_meter_filt': datum.pos_x_meter_filt,
            'pos_y_meter_filt': datum.pos_y_meter_filt,
            'abs_speed_mPs': datum.abs_speed_mPs,
            'tortuosity': datum.tortuosity
        } for datum in trajectory_data])
        
        if output_file:
            # Write to file
            data.to_csv(output_file, index=False)
            print(f"Data written to {output_file}")
        else:
            # Print to standard output
            print(data)

        session.close()

    def make_session(self):
        """
        Creates and returns a new SQLAlchemy session.

        This method initializes a sessionmaker with the current database engine, then invokes it to create a new session. The session is bound to the engine specified at the creation of the DeerDatabaseHandler instance, facilitating transactions and queries within the database context defined by the engine.

        Returns
        -------
        sqlalchemy.orm.session.Session
            A new SQLAlchemy session object that can be used to interact with the database.
            """
            return sessionmaker(bind=self.engine)()

    def insert_video_observation_data(self, csv_file_path, fps):
        """
        Inserts video observation data from a CSV file into the video_observation_reference table.
        
        Parameters
        ----------
        csv_file_path : str
            The path to the CSV file containing video observation data.
        fps : int
            Frames per second, used to calculate the stop_time of each video.
            """
            df = pd.read_csv(csv_file_path)
            session = self.make_session()

        for index, row in df.iterrows():
            video_duration = row['Frame_Count'] / fps
            start_time = datetime.datetime.strptime(row['Start_Time'], '%Y-%m-%d %H:%M:%S')
            stop_time = start_time + datetime.timedelta(seconds=video_duration)

            video_observation = VideoObservationReference(
                original_file_path=row['Filepath'],
                repetition=row['Rep'],
                deer=row['Deer'],
                frame_count=row['Frame_Count'],
                start_time=start_time,
                stop_time=stop_time,
                comment=row['Comments'] if 'Comments' in row else None
            )
            session.add(video_observation)

        session.commit()
        session.close()

    def generate_video_availability_csv(self, deer_number, repetition_number, output_dir):
        """
        Generates a CSV file containing data for the video_availability table,
        based on accelerometer data falling within the video observation timeframes.

        Parameters
        ----------
        deer_number : int
            The deer identifier.
        repetition_number : int
            The repetition sequence identifier.
        output_dir : str
            Directory where the output CSV will be saved.
            """
            session = self.make_session()

        # Fetch accelerometer data
        acc_data = session.query(AccelerometerData).join(DeerInfo).filter(
            DeerInfo.deer_number == f'D{deer_number}',
            DeerInfo.repetition_number == f'R{repetition_number}'
        ).all()

        # Fetch video observations
        video_obs = session.query(VideoObservationReference).filter(
            VideoObservationReference.deer == deer_number,
            VideoObservationReference.repetition == repetition_number
        ).all()

        output_data = []

        for acc in tqdm(acc_data, desc=f'vid check D{deer_number} R{repetition_number}'):
            for video in video_obs:
                if video.start_time <= acc.NZ_DateTime <= video.stop_time:
                    # Calculate frame number
                    time_diff = (acc.NZ_DateTime - video.start_time).total_seconds()
                    frame_number = int(time_diff * 30)  # Assuming 30 FPS

                    output_data.append({
                        "accelerometer_data_id": acc.data_id,
                        "video_observation_reference_id": video.id,
                        "frame": frame_number
                    })
                    break  # Move to the next accelerometer data point

        # Convert to DataFrame and write to CSV
        output_df = pd.DataFrame(output_data)
        output_path = f"{output_dir}/video_availability_D{deer_number}_R{repetition_number}.csv"
        output_df.to_csv(output_path, index=False)

        session.close()

    def import_video_availability_from_csv(self, csv_file_path):
        """
        Reads a CSV file and inserts its content into the video_availability table.

        Parameters
        ----------
        csv_file_path : str
            The path to the CSV file containing video availability data.
            """
            # Initialize a session
            session = self.make_session()

        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Iterate through the DataFrame rows
        for index, row in df.iterrows():
            try:
                # Create a new VideoAvailability instance for each row
                video_availability_entry = VideoAvailability(
                    accelerometer_data_id=row['accelerometer_data_id'],
                    video_observation_reference_id=row['video_observation_reference_id'] if not pd.isna(row['video_observation_reference_id']) else None,
                    frame=row['frame'] if not pd.isna(row['frame']) else None
                )
                # Add the new instance to the session
                session.add(video_availability_entry)
            
            except SQLAlchemyError as e:
                print(f"An error occurred: {e}")
                session.rollback()  # Rollback the transaction on error
                
        # Commit the transaction
        try:
            session.commit()
        except SQLAlchemyError as e:
            print(f"Commit failed, rolling back. Error: {e}")
            session.rollback()
        finally:
            session.close()

        print("Data import completed.")

    def insert_cluster_labels_from_npy(self, npy_file_path):
        """
        Reads cluster labels from a .npy file and inserts them into the cluster_labels table,
        associating each label with the correct accelerometer_data_id from the database.

        Parameters
        ----------
        npy_file_path : str
            The file path to the .npy file containing cluster labels.
            """
            # Load the labels from the .npy file
            labels = np.load(npy_file_path)

        session = self.make_session()

        # Fetch all data_id values from the accelerometer_data table
        accelerometer_data_ids = session.query(AccelerometerData.data_id).order_by(AccelerometerData.data_id).all()
        accelerometer_data_ids = [data_id[0] for data_id in accelerometer_data_ids]  # Convert list of tuples to list of ints

        # Make sure we have as many labels as we have accelerometer entries
        if len(labels) != len(accelerometer_data_ids):
            print("Warning: The number of labels does not match the number of accelerometer data entries.")
            # Handle this discrepancy according to your needs

        for data_id, label in tqdm(zip(accelerometer_data_ids, labels),desc='inserting labels'):
            cluster_label_entry = ClusterLabels(data_id=data_id, label=int(label))
            session.add(cluster_label_entry)

        session.commit()
        session.close()
        print("Cluster labels imported successfully.")



    def calculate_statistics_for_cluster(self, label, column_name):
        """
        Calculates the mean and standard error of the mean (SEM) for a specified column
        in the trajectory_data table associated with a specific cluster label.

        Parameters
        ----------
        label : int
            The cluster label to filter the data by.
        column_name : str
            The name of the column ('abs_speed_mPs' or 'tortuosity') for which to calculate the statistics.

        Returns
        -------
        tuple
            A tuple containing the mean and SEM for the specified column and cluster label.
            """
            session = self.make_session()

        # Ensure the column name is valid
        if column_name not in ['abs_speed_mPs', 'tortuosity']:
            print("Invalid column name. Please choose 'abs_speed_mPs' or 'tortuosity'.")
            return None

        try:
            # Query the database for the specified column values associated with the given cluster label
            query = session.query(TrajectoryData).join(ClusterLabels, TrajectoryData.data_id == ClusterLabels.data_id).filter(ClusterLabels.label == label)
            values = [getattr(record, column_name) for record in query.all() if getattr(record, column_name) is not None]

            if values:
                # Calculate the mean and SEM
                mean_val = np.mean(values)
                sem_val = stats.sem(values)

                print(f"Mean {column_name} for cluster {label}: {mean_val}")
                print(f"SEM {column_name} for cluster {label}: {sem_val}")

                return (mean_val, sem_val)
            else:
                print(f"No data found for cluster label {label}.")

        except SQLAlchemyError as e:
            print(f"An error occurred: {e}")
        finally:
            session.close()

        return None



    def update_json_with_statistics(self,json_file_path, column_name):
        """
        Reads a JSON file, updates it with the mean and SEM for the specified column,
        and saves the changes back to the JSON file.

        Parameters
        ----------
        json_file_path : str
            Path to the JSON file to update.
        column_name : str
            Name of the column to calculate statistics for.
        handler : DeerDatabaseHandler
            Instance of DeerDatabaseHandler to use for database queries.
            """
            with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        centroids = data.get("centroids", [])
        
        for centroid in centroids:
            cluster_label = centroid["centroid"]
            mean, sem = self.calculate_statistics_for_cluster(cluster_label, column_name)
            
            # Update the centroid dictionary with the new values
            centroid[f"{column_name}_mean"] = mean
            centroid[f"{column_name}_sem"] = sem

        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <cluster_label> <column_name> <json_file_path>")
        sys.exit(1)

    _, column_name, json_file_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # Initialize the database handler
    database_url = "sqlite:////projects/sciences/zoology/geurten_lab/deer_2024/deer_data_gps.db"  # Adjust as needed
    #database_url = 'sqlite:////home/geuba03p/deer_accl/deer_data_gps.db'
    handler = DeerDatabaseHandler(database_url)
    handler.create_database()  # Make sure the database is created and populated with data

    # Update the JSON file with statistics
    handler.update_json_with_statistics(json_file_path, column_name, handler)


# if __name__ == '__main__':
#     database_url = "sqlite:////projects/sciences/zoology/geurten_lab/deer_2024/deer_data_gps.db"  # Adjust as needed
#     #database_url = 'sqlite:////home/geuba03p/deer_accl/deer_data_gps.db'
#     handler = DeerDatabaseHandler(database_url)
#     handler.create_database()  # Make sure the database is created and populated with data


    # #npy_file = '/home/geuba03p/deer_cluster/deer5_2_labels_k7_delSize0_delPosP25.npy'
    # npy_file = '/projects/sciences/zoology/geurten_lab/deer_2024/cluster_results/deer5_2/delSize_0/k_7/labels/deer5_2_labels_k7_delSize0_delPosP25.npy'

    # handler.insert_cluster_labels_from_npy(npy_file)

    # directory_path = '/projects/sciences/zoology/geurten_lab/deer_2024/vid_avail_cache/'
    # # List all CSV files in the directory
    # for filename in tqdm(os.listdir(directory_path),desc='read csv files:'):
    #     if filename.endswith(".csv"):
    #         full_path = os.path.join(directory_path, filename)
    #         print(f"Importing {full_path} into database...")
    #         handler.import_video_availability_from_csv(full_path)
    #         print("Import completed.")


    # if len(sys.argv) != 4:
    #     print("Usage: python this_script.py <deer_number> <repetition_number> <output_directory>")
    #     sys.exit(1)

    # # Parse deer number and repetition number from command line
    # deer_number = int(sys.argv[1])
    # repetition_number = int(sys.argv[2])
    # output_directory = sys.argv[3]

    # database_url = "sqlite:////projects/sciences/zoology/geurten_lab/deer_2024/deer_data_gps.db"  # Adjust as needed
    # handler = DeerDatabaseHandler(database_url)
    # handler.create_database()  # Make sure the database is created and populated with data

    # # Call the function with command-line arguments
    # handler.generate_video_availability_csv(deer_number, repetition_number, output_directory)

# if __name__ == '__main__':
#     database_url = "sqlite:////home/geuba03p/deer_accl/deer_data_gps.db"  # Adjust as needed
#     handler = DeerDatabaseHandler(database_url)
#     handler.create_database()  # Make sure the database is created and populated with data
#     session = handler.make_session()
#     deer_codes = session.query(DeerInfo.deer_number, DeerInfo.repetition_number).distinct().all()

#     with open('./Deer_Codes_fromDB.csv', 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         for deer_code in deer_codes:
#             writer.writerow(deer_code)

#     session.close()