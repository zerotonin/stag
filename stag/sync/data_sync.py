"""
Sensor data synchronisation for head and ear accelerometers.

This module provides the :class:`BetterDataSync` class, which aligns
tri-axial accelerometer streams recorded on the head and ear of a deer
by detecting calibration-drop events (three controlled 1.5 m drops
recorded simultaneously by both loggers).
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from stag.sync.utils import (
    correct_calibration,
    get_calibrated_absolute_accelleration,
    get_consecutive_differences,
    make_absolute,
    sum_columns,
)
from stag.utils.csv_formatter import CsvFormatter


class BetterDataSync:
    """Synchronise head and ear accelerometer data via calibration drops.

    Parameters
    ----------
    deer_id : str
        Identifier for the deer (e.g. ``"R1_D1"``).
    head_data : pandas.DataFrame
        Accelerometer data from the head-mounted logger with columns
        ``'X'``, ``'Y'``, ``'Z'``.
    ear_data : pandas.DataFrame
        Accelerometer data from the ear-mounted logger.
    window_dict : dict
        Processing window with keys ``'start'`` and ``'end'`` (sample
        indices).
    log : bool, optional
        Enable CSV logging. Default ``True``.
    log_folder : str, optional
        Directory for log files.
    mkplot : bool, optional
        Generate diagnostic plots. Default ``False``.
    plot_folder : str, optional
        Directory for saved plots.

    Attributes
    ----------
    drops_dict : dict
        Detected calibration-drop timestamps after synchronisation.
    """

    def __init__(
        self,
        deer_id,
        head_data,
        ear_data,
        window_dict,
        log=True,
        log_folder="",
        mkplot=False,
        plot_folder="",
    ):
        self.deer_id = deer_id
        self.head_data = head_data
        self.ear_data = ear_data
        self.window_dict = window_dict
        self.drops_dict = {}
        self.mkplot = mkplot
        self.log = log
        self.log_folder = log_folder
        self.plot_folder = plot_folder

        if self.mkplot and len(self.plot_folder) == 0:
            print("Cannot save plots because no plot folder has been specified.")
            self.mkplot = False

        if self.log and len(self.log_folder) == 0:
            print("Cannot log because no log folder has been specified.")
            self.log = False

        if self.log:
            self._setup_logger()

    def _setup_logger(self):
        """Configure a CSV-formatted file logger."""
        self.logger = logging.getLogger(f"sync_{self.deer_id}")
        self.logger.setLevel(logging.DEBUG)
        log_path = os.path.join(self.log_folder, f"{self.deer_id}_sync.csv")
        handler = logging.FileHandler(log_path, mode="w")
        handler.setFormatter(CsvFormatter())
        self.logger.addHandler(handler)

    def _preprocess(self, data, columns=None):
        """Z-score, take absolute values, and sum columns.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw accelerometer data.
        columns : list of str, optional
            Column names to process. Defaults to ``['X', 'Y', 'Z']``.

        Returns
        -------
        pandas.Series
            Summed absolute z-scored acceleration.
        """
        if columns is None:
            columns = ["X", "Y", "Z"]
        calibrated = correct_calibration(data, columns)
        absolute = make_absolute(calibrated)
        return sum_columns(absolute)

    def detect_drops(self, signal, prominence=5.0, distance=500):
        """Detect calibration-drop peaks in a preprocessed signal.

        Parameters
        ----------
        signal : array-like
            Preprocessed (summed absolute z-scored) acceleration signal.
        prominence : float, optional
            Minimum peak prominence. Default ``5.0``.
        distance : int, optional
            Minimum samples between peaks. Default ``500``.

        Returns
        -------
        numpy.ndarray
            Indices of detected peaks.
        """
        peaks, _ = find_peaks(signal, prominence=prominence, distance=distance)
        return peaks

    def run_synchronization(self):
        """Execute the full synchronisation pipeline.

        Returns
        -------
        dict or None
            Dictionary with ``'head'`` and ``'ear'`` drop indices if
            successful, ``None`` otherwise.
        """
        head_signal = self._preprocess(self.head_data)
        ear_signal = self._preprocess(self.ear_data)

        head_peaks = self.detect_drops(head_signal)
        ear_peaks = self.detect_drops(ear_signal)

        if len(head_peaks) < 3 or len(ear_peaks) < 3:
            msg = (
                f"{self.deer_id}: fewer than 3 drops detected "
                f"(head={len(head_peaks)}, ear={len(ear_peaks)})"
            )
            if self.log:
                self.logger.warning(msg)
            print(msg)
            return None

        if self.log:
            self.logger.info(
                f"Drops detected — head: {head_peaks[:3]}, ear: {ear_peaks[:3]}"
            )

        self.drops_dict = {
            "head": head_peaks[:3].tolist(),
            "ear": ear_peaks[:3].tolist(),
        }
        return self.drops_dict
