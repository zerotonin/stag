"""
Utility functions for accelerometer data preprocessing.

Helper functions used during sensor synchronisation, including
z-score calibration, absolute-value transforms, column summation,
and consecutive-difference computation.
"""

import pandas as pd


def correct_calibration(data, cols=None):
    """Z-score the specified columns (zero mean, unit variance).

    Parameters
    ----------
    data : pandas.DataFrame
        Input accelerometer data.
    cols : list of str, optional
        Columns to standardise. Default ``['X', 'Y', 'Z']``.

    Returns
    -------
    pandas.DataFrame
        Z-scored copy of the selected columns.
    """
    if cols is None:
        cols = ["X", "Y", "Z"]
    subset = data[cols].copy()
    subset = (subset - subset.mean(skipna=True)) / subset.std(skipna=True)
    return subset


def make_absolute(data):
    """Return the element-wise absolute value of a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with absolute values.
    """
    return data.abs()


def sum_columns(data):
    """Sum all columns row-wise.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data.

    Returns
    -------
    pandas.Series
        Row-wise sum.
    """
    return data.sum(axis=1)


def get_consecutive_differences(series):
    """Compute first-order differences of a Series.

    Parameters
    ----------
    series : pandas.Series
        Input time series.

    Returns
    -------
    pandas.Series
        Consecutive differences (length = original − 1).
    """
    return series.diff().dropna()


def get_calibrated_absolute_accelleration(data, cols=None):
    """One-step pipeline: z-score → absolute → sum.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw accelerometer data.
    cols : list of str, optional
        Columns to process. Default ``['X', 'Y', 'Z']``.

    Returns
    -------
    pandas.Series
        Summed absolute z-scored acceleration.
    """
    if cols is None:
        cols = ["X", "Y", "Z"]
    calibrated = correct_calibration(data, cols)
    absolute = make_absolute(calibrated)
    return sum_columns(absolute)
