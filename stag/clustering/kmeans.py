"""
GPU-accelerated k-means clustering with contiguous leave-out stability.

This module implements the core clustering stage of the STAG pipeline.
It partitions z-scored accelerometer feature vectors into *k* prototypical
movements using RAPIDS cuML *k*-means on GPU, evaluates cluster quality
via the Calinski--Harabasz index, and supports a contiguous leave-out
scheme for robustness analysis.

The script is designed for SLURM array-job submission: all parameters
(k, deletion size, deletion position, random state) are accepted as
command-line arguments.

Example
-------
.. code-block:: bash

    python -m stag.clustering.kmeans \\
        -t deer8 -nc 8 -ds 0 -dp 0 -rs 0 \\
        -df data/clust_data.npy -sd results/
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

try:
    import cupy as cp
    from cuml.cluster import KMeans
    from cuml.preprocessing import StandardScaler

    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False

from sklearn.metrics import calinski_harabasz_score


def shrink_data(data, reduction_percent, cut_position_percent):
    """Remove a contiguous block from the data for stability analysis.

    Implements the circular leave-out scheme described in the paper:
    a block of ``reduction_percent`` % of the data starting at
    ``cut_position_percent`` % is excised, wrapping around if the block
    extends past the end of the array.

    Parameters
    ----------
    data : numpy.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    reduction_percent : float
        Percentage of data to remove (0–100).
    cut_position_percent : float
        Starting position of the cut as a percentage of total length.

    Returns
    -------
    numpy.ndarray
        Reduced feature matrix.
    """
    total_size = len(data)
    cut_size = int(total_size * (reduction_percent / 100.0))
    cut_start = int(total_size * (cut_position_percent / 100.0))
    cut_end = cut_start + cut_size

    if cut_end > total_size:
        overspill = cut_end - total_size
        data = np.delete(data, np.s_[cut_start:total_size], axis=0)
        overspill_adjusted = overspill - (total_size - len(data))
        data = np.delete(data, np.s_[:overspill_adjusted], axis=0)
    else:
        data = np.delete(data, np.s_[cut_start:cut_end], axis=0)

    return data


def generate_filename(parent_dir, tag, num_clusters, deletion_size, deletion_position):
    """Build standardised output paths for centroids, labels, and metadata.

    Parameters
    ----------
    parent_dir : str
        Root directory for results.
    tag : str
        Experiment tag (e.g. ``"deer8"``).
    num_clusters : int
        Number of clusters (*k*).
    deletion_size : int
        Deletion size percentage.
    deletion_position : int
        Deletion position percentage.

    Returns
    -------
    dict
        Dictionary with keys ``'centroids'``, ``'labels'``, ``'meta'``
        mapping to their respective file paths.
    """
    base = os.path.join(parent_dir, tag, f"delSize_{deletion_size}", f"k_{num_clusters}")
    filenames = {}
    for ftype, ext in [("centroids", "npy"), ("labels", "npy"), ("meta", "json")]:
        subdir = os.path.join(base, ftype) if ftype in ("centroids", "labels") else base
        os.makedirs(subdir, exist_ok=True)
        fname = (
            f"{tag}_{ftype}_k{num_clusters}_delSize{deletion_size}"
            f"_delPosP{deletion_position}.{ext}"
        )
        filenames[ftype] = os.path.join(subdir, fname)
    return filenames


def save_output(
    centroids, labels, quality_score, data_file, reduction_percent,
    cut_position_percent, filenames, start_time, duration,
):
    """Persist clustering results (centroids, labels, metadata JSON).

    Parameters
    ----------
    centroids : numpy.ndarray
        Cluster centroids, shape ``(k, n_features)``.
    labels : numpy.ndarray
        Per-sample cluster assignments.
    quality_score : float
        Calinski--Harabasz index.
    data_file : str
        Path to the input data file.
    reduction_percent : float
        Deletion size used.
    cut_position_percent : float
        Deletion position used.
    filenames : dict
        Output paths from :func:`generate_filename`.
    start_time : datetime.datetime
        Analysis start timestamp.
    duration : datetime.timedelta
        Wall-clock duration of the analysis.
    """
    np.save(filenames["centroids"], centroids)
    np.save(filenames["labels"], labels)
    metadata = {
        "calinski_harabasz_score": quality_score,
        "data_file": data_file,
        "reduction_percent": reduction_percent,
        "cut_position_percent": cut_position_percent,
        "centroids": centroids.tolist(),
        "analysis_start_date": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_duration": str(duration),
    }
    with open(filenames["meta"], "w") as f:
        json.dump(metadata, f)


def get_quality(labels, data_gpu_scaled):
    """Compute the Calinski--Harabasz index for a clustering solution.

    Parameters
    ----------
    labels : numpy.ndarray
        Cluster assignments.
    data_gpu_scaled : cupy.ndarray or numpy.ndarray
        Standardised feature matrix (on GPU or CPU).

    Returns
    -------
    float
        Calinski--Harabasz score, or ``NaN`` if only one cluster is
        populated.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        if _GPU_AVAILABLE:
            data_cpu = cp.asnumpy(data_gpu_scaled)
        else:
            data_cpu = np.asarray(data_gpu_scaled)
        return calinski_harabasz_score(data_cpu, labels)
    return np.nan


def main(tag, n_clusters, deletion_size, deletion_position, random_state, data_file, save_dir):
    """Run a single k-means clustering job.

    Parameters
    ----------
    tag : str
        Experiment tag.
    n_clusters : int
        Number of clusters.
    deletion_size : int
        Percentage of contiguous data to leave out (0 = full data).
    deletion_position : int
        Starting position of the leave-out block (percentage).
    random_state : int
        Random seed for k-means initialisation.
    data_file : str
        Path to the ``.npy`` feature matrix.
    save_dir : str
        Root directory for output files.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError(
            "RAPIDS cuML is not installed. GPU clustering requires "
            "cupy and cuml — see environment.yml for installation."
        )

    start_time = datetime.datetime.now()

    arraydata = np.load(data_file)
    arraydata = shrink_data(arraydata, deletion_size, deletion_position)

    data_gpu = cp.asarray(arraydata)
    scaler = StandardScaler()
    data_gpu_scaled = scaler.fit_transform(data_gpu)

    filenames = generate_filename(save_dir, tag, n_clusters, deletion_size, deletion_position)

    print(f"{datetime.datetime.now()} Starting KMeans clustering (k={n_clusters})")
    kmeans = KMeans(init="k-means||", n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(data_gpu_scaled)

    labels = kmeans.labels_.get()
    centroids = kmeans.cluster_centers_.get()

    quality_score = get_quality(labels, data_gpu_scaled)
    duration = datetime.datetime.now() - start_time

    save_output(
        centroids, labels, quality_score, data_file,
        deletion_size, deletion_position, filenames, start_time, duration,
    )
    print(f"Done — CH = {quality_score:.1f}, duration = {duration}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="STAG k-means clustering with contiguous leave-out stability."
    )
    parser.add_argument("--external", action="store_true",
                        help="Print the metadata filename and exit.")
    parser.add_argument("-t", "--tag", type=str, help="Experiment tag.")
    parser.add_argument("-nc", "--n_clusters", type=int, help="Number of clusters.")
    parser.add_argument("-ds", "--deletion_size", type=int, help="Deletion size %%.")
    parser.add_argument("-dp", "--deletion_position", type=int, help="Deletion position %%.")
    parser.add_argument("-rs", "--random_state", type=int, default=0, help="Random seed.")
    parser.add_argument("-df", "--data_file_position", type=str, help="Path to .npy data.")
    parser.add_argument("-sd", "--save_dir", type=str, default="./", help="Output directory.")

    args = parser.parse_args()

    if args.external:
        if not all([args.tag, args.n_clusters is not None,
                    args.deletion_size is not None, args.deletion_position is not None]):
            print("Missing required arguments for filename generation.")
            sys.exit(1)
        filenames = generate_filename(
            args.save_dir, args.tag, args.n_clusters,
            args.deletion_size, args.deletion_position,
        )
        print(filenames["meta"])
    else:
        required = [args.tag, args.n_clusters, args.deletion_size,
                    args.deletion_position, args.data_file_position]
        if any(arg is None for arg in required):
            parser.print_help()
            sys.exit(1)
        main(
            args.tag, args.n_clusters, args.deletion_size,
            args.deletion_position, args.random_state,
            args.data_file_position, args.save_dir,
        )
