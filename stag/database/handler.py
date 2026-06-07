# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — database.handler                                         ║
# ║  « create, ingest, and query the deer sensor database »          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  DeerDatabaseHandler owns the SQLAlchemy engine and provides    ║
# ║  the ingestion + query surface for the schema defined in        ║
# ║  stag.database.orm.  Methods are grouped by concern:            ║
# ║                                                                  ║
# ║    Schema / sessions:  create_database, make_session            ║
# ║    Sensor ingestion:   read_h5_file, insert_data_from_directory,║
# ║                        insert_trajectory_data_from_h5           ║
# ║    Identity lookup:    get_deer_id_from_filename                ║
# ║    Trajectory export:  write_trajectory_data_for_deer           ║
# ║    Video alignment:    insert_video_observation_data,           ║
# ║                        generate_video_availability_csv,         ║
# ║                        import_video_availability_from_csv       ║
# ║    Cluster labels:     insert_cluster_labels_from_npy           ║
# ║    Per-cluster stats:  calculate_statistics_for_cluster,        ║
# ║                        update_json_with_statistics              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Database handler — ingestion and query methods for the deer schema."""

from __future__ import annotations

import csv
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from stag.database.orm import (
    AccelerometerData,
    Base,
    ClusterLabels,
    DeerInfo,
    TrajectoryData,
    VideoAvailability,
    VideoObservationReference,
)


class DeerDatabaseHandler:
    """Engine-owning handler for the STAG deer sensor database.

    Attributes:
        engine: SQLAlchemy ``Engine`` bound to the configured database URL.
    """

    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url)

    # ─────────────────────────────────────────────────────────────────
    #  Schema and session management
    # ─────────────────────────────────────────────────────────────────

    def create_database(self) -> None:
        """Create the database schema (idempotent — uses ``create_all``)."""
        Base.metadata.create_all(self.engine)

    def make_session(self):
        """Return a new SQLAlchemy session bound to ``self.engine``."""
        return sessionmaker(bind=self.engine)()

    # ─────────────────────────────────────────────────────────────────
    #  Sensor ingestion (HDF5 → DB)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def read_h5_file(filename: str) -> pd.DataFrame:
        """Read the ``'df'`` table from an HDF5 file."""
        return pd.read_hdf(filename, "df")

    def insert_data_from_directory(self, directory: str) -> None:
        """Insert accelerometer rows from every ``*.h5`` in ``directory``.

        Filenames are expected to follow the pattern ``*_R<rep>_D<deer>*.h5``.
        """
        session = self.make_session()

        for filename in os.listdir(directory):
            if not filename.endswith(".h5"):
                continue

            parts = filename.split("_")
            repetition, deer = parts[1], parts[2]
            data = self.read_h5_file(os.path.join(directory, filename))

            deer_info = DeerInfo(repetition_number=repetition, deer_number=deer)
            session.add(deer_info)
            session.flush()

            for _, row in data.iterrows():
                session.add(
                    AccelerometerData(
                        deer_id=deer_info.deer_id,
                        X_head=row["X_head"], Y_head=row["Y_head"], Z_head=row["Z_head"],
                        X_ear=row["X_ear"], Y_ear=row["Y_ear"], Z_ear=row["Z_ear"],
                    )
                )

            session.commit()

        session.close()

    def insert_trajectory_data_from_h5(
        self, file_position: str, data: pd.DataFrame,
    ) -> None:
        """Insert trajectory rows for the deer identified by the filename."""
        session = self.make_session()

        if not file_position.endswith(".h5"):
            session.close()
            return

        deer_id = self.get_deer_id_from_filename(os.path.basename(file_position))
        if deer_id is None:
            session.close()
            raise ValueError(
                f"No deer matching {file_position!r}; insert accelerometer "
                "data or register the animal first.",
            )

        for _, row in data.iterrows():
            session.add(
                TrajectoryData(
                    deer_id=deer_id,
                    pos_WGS84_lat=row["location-lat"],
                    pos_WGS84_lon=row["location-lon"],
                    pos_NZMG_x_meter=row["pos_x_meter"],
                    pos_NZMG_y_meter=row["pos_y_meter"],
                    pos_x_meter_filt=row["pos_x_meter_filt"],
                    pos_y_meter_filt=row["pos_y_meter_filt"],
                    abs_speed_mPs=row["abs_speed_mPs"],
                    tortuosity=row["tortuosity"],
                )
            )

        session.commit()
        session.close()

    # ─────────────────────────────────────────────────────────────────
    #  Identity lookup
    # ─────────────────────────────────────────────────────────────────

    def get_deer_id_from_filename(self, filename: str) -> int | None:
        """Return ``deer_id`` for a filename of pattern ``*_R<rep>_D<deer>*.h5``."""
        session = self.make_session()

        parts = filename.split("_")
        repetition_part = next((p for p in parts if p.startswith("R")), None)
        deer_part = next((p for p in parts if p.startswith("D")), None)

        if repetition_part is None or deer_part is None:
            session.close()
            return None

        deer_info = (
            session.query(DeerInfo)
            .filter_by(repetition_number=repetition_part, deer_number=deer_part)
            .first()
        )
        session.close()
        return deer_info.deer_id if deer_info else None

    # ─────────────────────────────────────────────────────────────────
    #  Trajectory export
    # ─────────────────────────────────────────────────────────────────

    def write_trajectory_data_for_deer(
        self, deer_number: int, output_file: str | None = None,
    ) -> None:
        """Write a deer's trajectory rows to CSV (or stdout if ``output_file`` is None)."""
        session = self.make_session()

        deer_info = session.query(DeerInfo).filter_by(deer_id=deer_number).first()
        if not deer_info:
            print(f"No data found for deer_number: {deer_number}")
            session.close()
            return

        trajectory_data = (
            session.query(TrajectoryData).filter_by(deer_id=deer_info.deer_id).all()
        )

        data = pd.DataFrame([{
            "data_id": d.data_id,
            "pos_WGS84_lat": d.pos_WGS84_lat,
            "pos_WGS84_lon": d.pos_WGS84_lon,
            "pos_NZMG_x_meter": d.pos_NZMG_x_meter,
            "pos_NZMG_y_meter": d.pos_NZMG_y_meter,
            "pos_x_meter_filt": d.pos_x_meter_filt,
            "pos_y_meter_filt": d.pos_y_meter_filt,
            "abs_speed_mPs": d.abs_speed_mPs,
            "tortuosity": d.tortuosity,
        } for d in trajectory_data])

        if output_file:
            data.to_csv(output_file, index=False)
            print(f"Data written to {output_file}")
        else:
            print(data)

        session.close()

    # ─────────────────────────────────────────────────────────────────
    #  Video observation alignment
    # ─────────────────────────────────────────────────────────────────

    def insert_video_observation_data(self, csv_file_path: str, fps: int) -> None:
        """Insert one VideoObservationReference row per CSV row."""
        df = pd.read_csv(csv_file_path)
        session = self.make_session()

        for _, row in df.iterrows():
            duration = row["Frame_Count"] / fps
            start_time = datetime.datetime.strptime(
                row["Start_Time"], "%Y-%m-%d %H:%M:%S",
            )
            stop_time = start_time + datetime.timedelta(seconds=duration)

            session.add(
                VideoObservationReference(
                    original_file_path=row["Filepath"],
                    repetition=row["Rep"],
                    deer=row["Deer"],
                    frame_count=row["Frame_Count"],
                    start_time=start_time,
                    stop_time=stop_time,
                    comment=row["Comments"] if "Comments" in row else None,
                )
            )

        session.commit()
        session.close()

    def generate_video_availability_csv(
        self, deer_number: int, repetition_number: int, output_dir: str,
    ) -> None:
        """Emit CSV linking accelerometer rows to video frames for one (deer, rep)."""
        session = self.make_session()

        acc_data = (
            session.query(AccelerometerData)
            .join(DeerInfo)
            .filter(
                DeerInfo.deer_number == f"D{deer_number}",
                DeerInfo.repetition_number == f"R{repetition_number}",
            )
            .all()
        )
        video_obs = (
            session.query(VideoObservationReference)
            .filter(
                VideoObservationReference.deer == deer_number,
                VideoObservationReference.repetition == repetition_number,
            )
            .all()
        )

        output_data = []
        for acc in tqdm(acc_data, desc=f"vid check D{deer_number} R{repetition_number}"):
            for video in video_obs:
                if video.start_time <= acc.NZ_DateTime <= video.stop_time:
                    time_diff = (acc.NZ_DateTime - video.start_time).total_seconds()
                    frame_number = int(time_diff * 30)  # video FPS
                    output_data.append({
                        "accelerometer_data_id": acc.data_id,
                        "video_observation_reference_id": video.id,
                        "frame": frame_number,
                    })
                    break

        output_df = pd.DataFrame(output_data)
        output_path = (
            f"{output_dir}/video_availability_D{deer_number}_R{repetition_number}.csv"
        )
        output_df.to_csv(output_path, index=False)

        session.close()

    def import_video_availability_from_csv(self, csv_file_path: str) -> None:
        """Load a video_availability CSV into the database."""
        session = self.make_session()
        df = pd.read_csv(csv_file_path)

        for _, row in df.iterrows():
            try:
                session.add(
                    VideoAvailability(
                        accelerometer_data_id=row["accelerometer_data_id"],
                        video_observation_reference_id=(
                            row["video_observation_reference_id"]
                            if not pd.isna(row["video_observation_reference_id"])
                            else None
                        ),
                        frame=row["frame"] if not pd.isna(row["frame"]) else None,
                    )
                )
            except SQLAlchemyError as e:
                print(f"An error occurred: {e}")
                session.rollback()

        try:
            session.commit()
        except SQLAlchemyError as e:
            print(f"Commit failed, rolling back. Error: {e}")
            session.rollback()
        finally:
            session.close()

        print("Data import completed.")

    # ─────────────────────────────────────────────────────────────────
    #  Cluster labels
    # ─────────────────────────────────────────────────────────────────

    def insert_cluster_labels_from_npy(self, npy_file_path: str) -> None:
        """Load cluster labels from a ``.npy`` file and link them to accel rows.

        Labels are assigned in ``data_id`` order; a length mismatch is
        logged but does not abort the insertion.
        """
        labels = np.load(npy_file_path)
        session = self.make_session()

        ids_query = (
            session.query(AccelerometerData.data_id)
            .order_by(AccelerometerData.data_id)
            .all()
        )
        ids = [row[0] for row in ids_query]

        if len(labels) != len(ids):
            print(
                "Warning: The number of labels does not match the number of "
                "accelerometer data entries.",
            )

        for data_id, label in tqdm(zip(ids, labels), desc="inserting labels"):
            session.add(ClusterLabels(data_id=data_id, label=int(label)))

        session.commit()
        session.close()
        print("Cluster labels imported successfully.")

    # ─────────────────────────────────────────────────────────────────
    #  Per-cluster statistics
    # ─────────────────────────────────────────────────────────────────

    def calculate_statistics_for_cluster(
        self, label: int, column_name: str,
    ) -> tuple[float, float] | None:
        """Return (mean, SEM) of a TrajectoryData column for one cluster label.

        Args:
            label:       Cluster label to filter on.
            column_name: Either ``'abs_speed_mPs'`` or ``'tortuosity'``.
        """
        if column_name not in ("abs_speed_mPs", "tortuosity"):
            print("Invalid column name. Please choose 'abs_speed_mPs' or 'tortuosity'.")
            return None

        session = self.make_session()
        try:
            query = (
                session.query(TrajectoryData)
                .join(ClusterLabels, TrajectoryData.data_id == ClusterLabels.data_id)
                .filter(ClusterLabels.label == label)
            )
            values = [
                getattr(record, column_name)
                for record in query.all()
                if getattr(record, column_name) is not None
            ]

            if not values:
                print(f"No data found for cluster label {label}.")
                return None

            mean_val = float(np.mean(values))
            sem_val = float(stats.sem(values))
            print(f"Mean {column_name} for cluster {label}: {mean_val}")
            print(f"SEM  {column_name} for cluster {label}: {sem_val}")
            return mean_val, sem_val

        except SQLAlchemyError as e:
            print(f"An error occurred: {e}")
            return None
        finally:
            session.close()

    def update_json_with_statistics(
        self, json_file_path: str, column_name: str,
    ) -> None:
        """Annotate a centroid JSON file with mean/SEM for one column."""
        with open(json_file_path, "r") as f:
            data = json.load(f)

        centroids = data.get("centroids", [])
        for centroid in centroids:
            cluster_label = centroid["centroid"]
            stats_pair = self.calculate_statistics_for_cluster(cluster_label, column_name)
            if stats_pair is not None:
                mean_val, sem_val = stats_pair
                centroid[f"{column_name}_mean"] = mean_val
                centroid[f"{column_name}_sem"] = sem_val

        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    from stag.local_paths import get_path

    if len(sys.argv) != 4:
        print(
            "Usage: python -m stag.database.handler "
            "<cluster_label> <column_name> <json_file_path>\n\n"
            "Resolves the SQLite URL via stag.local_paths "
            "(STAG_DEER_DB_URL env var, then deer_db_url in "
            "local_paths.json).",
        )
        sys.exit(1)

    _, column_name, json_file_path = sys.argv[1], sys.argv[2], sys.argv[3]
    database_url = get_path("deer_db_url")
    handler = DeerDatabaseHandler(database_url)
    handler.create_database()
    handler.update_json_with_statistics(json_file_path, column_name)
