# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — database.orm                                             ║
# ║  « SQLAlchemy ORM schema for the deer sensor database »          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Declarative-base classes that map the SQLite schema:           ║
# ║    DeerInfo                  — one row per (deer, repetition)   ║
# ║    AccelerometerData         — 50 Hz six-axis sensor stream     ║
# ║    TrajectoryData            — 0.5 Hz GPS + derived features    ║
# ║    VideoObservationReference — per-video clip metadata          ║
# ║    VideoAvailability         — frame-level video/accel linkage  ║
# ║    ClusterLabels             — per-sample cluster assignments   ║
# ║                                                                  ║
# ║  Importers should depend on this module, not on handler.py,     ║
# ║  to avoid pulling in pandas + tqdm just for the schema.         ║
# ╚══════════════════════════════════════════════════════════════════╝
"""SQLAlchemy ORM schema for the STAG deer sensor database."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class DeerInfo(Base):
    """One row per (repetition, deer) combination.

    Attributes:
        deer_id:            Primary key, autoincrementing.
        repetition_number:  Identifier for the data-collection repetition.
        deer_number:        Unique identifier for the deer.
        accelerometer_data: Reverse relationship to AccelerometerData.
        trajectory_data:    Reverse relationship to TrajectoryData.
    """

    __tablename__ = "deer_info"

    deer_id = Column(Integer, primary_key=True, autoincrement=True)
    repetition_number = Column(String(255), nullable=False)
    deer_number = Column(String(255), nullable=False)

    accelerometer_data = relationship("AccelerometerData", back_populates="deer_info")
    trajectory_data = relationship("TrajectoryData", back_populates="deer_info")

    __table_args__ = ({"mysql_charset": "utf8mb4"},)


class AccelerometerData(Base):
    """50 Hz six-axis accelerometer stream (three head, three ear axes)."""

    __tablename__ = "accelerometer_data"

    data_id = Column(Integer, primary_key=True, autoincrement=True)
    deer_id = Column(Integer, ForeignKey("deer_info.deer_id"))
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
    """0.5 Hz GPS positions (WGS84 + NZMG) plus derived speed and tortuosity."""

    __tablename__ = "trajectory_data"

    data_id = Column(Integer, primary_key=True, autoincrement=True)
    deer_id = Column(Integer, ForeignKey("deer_info.deer_id"))
    pos_WGS84_lat = Column(Float)
    pos_WGS84_lon = Column(Float)
    pos_NZMG_x_meter = Column(Float)
    pos_NZMG_y_meter = Column(Float)
    pos_x_meter_filt = Column(Float)
    pos_y_meter_filt = Column(Float)
    abs_speed_mPs = Column(Float)
    tortuosity = Column(Float)

    deer_info = relationship("DeerInfo", back_populates="trajectory_data")

    __table_args__ = ({"mysql_charset": "utf8mb4"},)


class VideoObservationReference(Base):
    """Ground-truth video clip metadata.

    The ``start_time`` / ``stop_time`` window is used to assign frame
    numbers to accelerometer samples via :class:`VideoAvailability`.
    """

    __tablename__ = "video_observation_reference"

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
    """Frame-level link between accelerometer samples and video observations.

    A NULL ``video_observation_reference_id`` or ``frame`` indicates that
    the accelerometer sample has no synchronised video coverage.
    """

    __tablename__ = "video_availability"

    id = Column(Integer, primary_key=True)
    accelerometer_data_id = Column(
        Integer, ForeignKey("accelerometer_data.data_id"),
    )
    video_observation_reference_id = Column(
        Integer, ForeignKey("video_observation_reference.id"), nullable=True,
    )
    frame = Column(Integer, nullable=True)

    accelerometer_data = relationship(
        "AccelerometerData", backref="video_availability",
    )
    video_observation_reference = relationship(
        "VideoObservationReference", backref="video_availability",
    )

    __table_args__ = ({"mysql_charset": "utf8mb4"},)


class ClusterLabels(Base):
    """Per-sample k-means cluster assignment."""

    __tablename__ = "cluster_labels"

    data_id = Column(
        Integer, ForeignKey("accelerometer_data.data_id"), primary_key=True,
    )
    label = Column(Integer, nullable=False)

    accelerometer_data = relationship(
        "AccelerometerData", backref="cluster_labels",
    )

    __table_args__ = ({"mysql_charset": "utf8mb4"},)
