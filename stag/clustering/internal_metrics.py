# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — clustering.internal_metrics                              ║
# ║  « Silhouette + Kneedle Elbow alongside CH »                     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Internal cluster-validation metrics — no labels required.       ║
# ║  Reviewer R2 asks for Silhouette and Elbow alongside the         ║
# ║  Calinski-Harabasz index that already drives Figure 2A.          ║
# ║                                                                  ║
# ║  Functions:                                                      ║
# ║    compute_silhouette_stratified  — silhouette on a per-cluster ║
# ║                                     stratified subsample so      ║
# ║                                     PM2/4/5 (<1 % of data) are   ║
# ║                                     adequately represented.      ║
# ║    compute_inertia                — within-cluster SSE = W(k).   ║
# ║    recompute_inertia_for_meta_dir — back-fill historical JSONs.  ║
# ║    locate_elbow_kneedle           — Kneedle algorithm on W(k).   ║
# ║    selection_summary              — per-k DataFrame ready for    ║
# ║                                     the revised Figure 2.        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Internal cluster-validation metrics: Silhouette, Inertia, Kneedle Elbow."""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# kneed (KneeLocator) is imported lazily inside locate_elbow_kneedle()
# so the rest of this module - in particular compute_silhouette_stratified
# and the inertia-recompute path used by SLURM array tasks - does not
# need kneed installed.  Kneed is only required when the elbow is
# actually located.
from sklearn.metrics import silhouette_samples
from tqdm import tqdm


def compute_inertia(
    X: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 1_000_000,
) -> float:
    """Within-cluster sum of squared distances — the k-means objective W(k).

    Streams over the feature matrix in ``chunk_size`` rows so the
    residual array never exceeds ``chunk_size × n_features × 8`` bytes
    of working memory (≈ 48 MB at the default).  This matters for
    parallel back-fill across many workers, where the naïve
    ``X - centroids[labels]`` would materialise a 13 GB residual per
    worker.

    Args:
        X:          Feature matrix, ``(n_samples, n_features)``.
        centroids:  Centroid matrix, ``(n_clusters, n_features)``.
        labels:     Per-sample cluster assignment, ``(n_samples,)``.
        chunk_size: Rows per streaming chunk.  Default 1e6.

    Returns:
        Scalar inertia.  Matches ``sklearn.cluster.KMeans.inertia_`` to
        floating-point tolerance when ``centroids`` and ``labels`` come
        from a converged fit.
    """
    n = X.shape[0]
    total = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        residuals = X[start:end] - centroids[labels[start:end]]
        total += float((residuals * residuals).sum())
    return total


def compute_silhouette_stratified(
    X: np.ndarray,
    labels: np.ndarray,
    n_per_cluster: int = 5000,
    n_repeats: int = 50,
    rng: np.random.Generator | None = None,
) -> dict[str, float | np.ndarray]:
    r"""Mean silhouette score via stratified per-cluster subsampling.

    Silhouette is :math:`\mathcal{O}(n^2)` in pairwise distances, so on
    the full ~$8.6\times10^{7}$ STAG sample it is infeasible.  This
    helper draws ``n_per_cluster`` samples from every cluster, computes
    silhouette on that subsample, and repeats ``n_repeats`` times with
    different seeds to give a median ± IQR.

    Stratification is essential: PM2/PM4/PM5 (the ear flicks) comprise
    < 1 % of the data.  Uniform subsampling (as in
    ``sklearn.silhouette_score(..., sample_size=N)``) would under-
    represent them and bias the metric.

    Args:
        X:             Feature matrix, ``(n_samples, n_features)``.
        labels:        Per-sample cluster assignment.
        n_per_cluster: Samples per cluster per repeat.  Default 5 000.
        n_repeats:     Number of independent subsamples.  Default 50.
        rng:           NumPy generator; created from default seed if None.

    Returns:
        Dict with keys:
          ``"mean_silhouette"``: median across repeats of the mean
              silhouette per subsample.
          ``"iqr_silhouette"``:  (lower, upper) quartiles across repeats.
          ``"per_repeat"``:      1-D array of per-repeat mean silhouettes.
          ``"per_cluster_mean"``: ``(n_clusters,)`` median silhouette per
              cluster across repeats.
    """
    if rng is None:
        rng = np.random.default_rng()

    unique = np.unique(labels)
    per_repeat_overall: list[float] = []
    per_repeat_per_cluster: list[np.ndarray] = []

    for _ in range(n_repeats):
        idx_parts: list[np.ndarray] = []
        for u in unique:
            members = np.where(labels == u)[0]
            take = min(n_per_cluster, members.size)
            idx_parts.append(rng.choice(members, size=take, replace=False))
        idx = np.concatenate(idx_parts)

        s_samples = silhouette_samples(X[idx], labels[idx])
        per_repeat_overall.append(float(s_samples.mean()))

        per_cluster = np.array([
            s_samples[labels[idx] == u].mean() for u in unique
        ])
        per_repeat_per_cluster.append(per_cluster)

    per_repeat_arr = np.array(per_repeat_overall)
    per_cluster_arr = np.array(per_repeat_per_cluster)

    return {
        "mean_silhouette": float(np.median(per_repeat_arr)),
        "iqr_silhouette": (
            float(np.quantile(per_repeat_arr, 0.25)),
            float(np.quantile(per_repeat_arr, 0.75)),
        ),
        "per_repeat": per_repeat_arr,
        "per_cluster_mean": np.median(per_cluster_arr, axis=0),
    }


def _recompute_inertia_one(
    args: tuple[str, str, bool, tuple[float, ...] | None],
) -> dict | None:
    """Worker for parallel inertia back-fill.

    Kept at module level so ProcessPoolExecutor can pickle it (per the
    house-style rule).  ``reduction_filter`` is an optional tuple of allowed
    ``reduction_percent`` values.  If supplied, fits with a different
    reduction_percent are skipped without touching their labels file.
    """
    jpath_str, data_path_str, overwrite, reduction_filter = args
    jpath = Path(jpath_str)
    data_path = Path(data_path_str)

    if "_meta_k" not in jpath.name:
        return None

    try:
        content = json.loads(jpath.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if "reduction_percent" not in content or "centroids" not in content:
        return None

    if reduction_filter is not None:
        if content["reduction_percent"] not in reduction_filter:
            return None

    try:
        centroids = np.array(content["centroids"], dtype=float)
    except (TypeError, ValueError):
        return None
    if centroids.ndim != 2:
        return None

    labels_path = _labels_path_from_meta(jpath, content)
    if labels_path is None or not labels_path.exists():
        return None

    labels = np.load(labels_path)

    # mmap the feature matrix so workers share the 13 GB pages via the
    # OS page cache instead of each materialising a copy.
    data = np.load(data_path, mmap_mode="r")

    # Replay shrink_data() to align X with labels.  shrink_data has a
    # fast path for reduction_percent == 0 that returns the mmap'd view
    # unchanged — critical for memory safety in parallel runs.
    from stag.clustering.kmeans import shrink_data
    X_used = shrink_data(
        data,
        content.get("reduction_percent", 0),
        content.get("cut_position_percent", 0),
    )

    if X_used.shape[0] != labels.shape[0]:
        return None

    inertia = compute_inertia(X_used, centroids, labels)

    if overwrite:
        content["inertia"] = inertia
        jpath.write_text(json.dumps(content))

    return {
        "file_path": str(jpath),
        "k_number": int(centroids.shape[0]),
        "reduction_percent": content.get("reduction_percent"),
        "cut_position_percent": content.get("cut_position_percent"),
        "inertia": inertia,
    }


def recompute_inertia_for_meta_dir(
    meta_dir: str | Path,
    data_path: str | Path,
    overwrite: bool = False,
    workers: int = 1,
    reduction_percents: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Back-fill ``inertia`` for every metadata JSON under ``meta_dir``.

    The pre-Sprint-0 ``save_output()`` did not persist ``inertia_``.
    This helper walks the JSON tree, locates the matching centroids
    and labels files, recomputes ``W(k)`` from the saved data, and
    optionally writes the value back into each JSON.

    Args:
        meta_dir:  Root directory containing the metadata JSON files.
        data_path: Path to the ``clust_data_*.npy`` feature matrix.
        overwrite: When True, edit each JSON in place to add the
                   ``"inertia"`` field.  Default False (returns a
                   DataFrame only).
        workers:            Process-pool size.  Default 1 (sequential).
                            Higher values speed up the back-fill
                            significantly when labels live on a local
                            SSD.  Workers share the feature matrix via
                            ``mmap_mode="r"`` so total RSS stays
                            bounded.
        reduction_percents: Optional whitelist of ``reduction_percent``
                            values.  When None (default), every fit is
                            back-filled.  Restricting to a single value
                            for a given run skips ~75 % of the fits
                            and matches the silhouette pass's filter.

    Returns:
        DataFrame indexed by JSON path with columns
        ``k_number``, ``reduction_percent``, ``cut_position_percent``,
        ``inertia``.
    """
    json_paths = sorted(Path(meta_dir).rglob("*.json"))
    filt = tuple(reduction_percents) if reduction_percents is not None else None
    worker_args = [(str(p), str(data_path), overwrite, filt) for p in json_paths]

    rows: list[dict] = []
    if workers <= 1:
        for args in tqdm(worker_args, desc="recompute inertia"):
            result = _recompute_inertia_one(args)
            if result is not None:
                rows.append(result)
    else:
        # chunksize tuned to amortise IPC over the per-fit work
        # (a single fit takes ~0.2-1 s on local SSD).
        chunksize = max(1, len(worker_args) // (workers * 8))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in tqdm(
                pool.map(_recompute_inertia_one, worker_args, chunksize=chunksize),
                total=len(worker_args),
                desc=f"recompute inertia [{workers}× parallel]",
            ):
                if result is not None:
                    rows.append(result)

    return pd.DataFrame(rows)


def _labels_path_from_meta(meta_path: Path, content: dict) -> Path | None:
    """Derive the labels.npy path from a meta JSON path.

    Layout (from :func:`stag.clustering.kmeans.generate_filename`):
      ``<root>/<tag>/delSize_<ds>/k_<k>/<tag>_meta_k<k>_delSize<ds>_delPosP<pos>.json``
      ``<root>/<tag>/delSize_<ds>/k_<k>/labels/<tag>_labels_k<k>_delSize<ds>_delPosP<pos>.npy``
    """
    name = meta_path.name
    if not name.endswith(".json") or "_meta_" not in name:
        return None
    base = name.replace("_meta_", "_labels_").replace(".json", ".npy")
    return meta_path.parent / "labels" / base


def locate_elbow_kneedle(
    k_values: Sequence[int],
    inertia: Sequence[float],
    S: float = 1.0,
    curve: str = "convex",
    direction: str = "decreasing",
) -> dict[str, float | int | None]:
    """Locate the elbow on a W(k) curve via the Kneedle algorithm.

    Wraps :class:`kneed.KneeLocator` (Satopää et al. 2011) and returns
    the chosen *k* plus diagnostic fields useful for the Figure 2C
    annotation.

    Args:
        k_values:  Monotonic increasing sequence of cluster counts.
        inertia:   ``W(k)`` values in the same order.  Should be
                   monotonically non-increasing.
        S:         Sensitivity parameter (lower → more aggressive
                   elbow detection).  Default 1.0.
        curve:     ``"convex"`` for inertia curves.  Default.
        direction: ``"decreasing"`` for inertia curves.  Default.

    Returns:
        Dict with:
          ``"elbow_k"``       — chosen k, or None if no elbow found.
          ``"elbow_y"``       — inertia at the elbow.
          ``"elbow_index"``   — position of the elbow in ``k_values``.
          ``"normalised_knee_distance"`` — Kneedle internal score
              (larger ⇒ more pronounced elbow).
    """
    from kneed import KneeLocator  # local import - see module-top comment

    kl = KneeLocator(
        list(k_values), list(inertia),
        S=S, curve=curve, direction=direction,
    )
    elbow_k = kl.knee
    if elbow_k is None:
        return {
            "elbow_k": None,
            "elbow_y": None,
            "elbow_index": None,
            "normalised_knee_distance": None,
        }
    idx = list(k_values).index(elbow_k)
    return {
        "elbow_k": int(elbow_k),
        "elbow_y": float(inertia[idx]),
        "elbow_index": idx,
        "normalised_knee_distance": (
            float(kl.knee_y) if kl.knee_y is not None else None
        ),
    }


def selection_summary(
    k_values: Sequence[int],
    ch: Sequence[float] | None = None,
    ch_low: Sequence[float] | None = None,
    ch_high: Sequence[float] | None = None,
    instability: Sequence[float] | None = None,
    instability_low: Sequence[float] | None = None,
    instability_high: Sequence[float] | None = None,
    silhouette: Sequence[float] | None = None,
    silhouette_low: Sequence[float] | None = None,
    silhouette_high: Sequence[float] | None = None,
    inertia: Sequence[float] | None = None,
    inertia_low: Sequence[float] | None = None,
    inertia_high: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Single per-k DataFrame ready for the revised Figure 2.

    Each ``metric`` is accompanied by an optional ``metric_low`` and
    ``metric_high`` aligned to ``k_values`` that the plotter renders
    as a 95 % confidence band around the median line.  Bounds are
    optional — missing bounds appear as NaN columns and the plotter
    silently skips the fill.

    Args:
        k_values:         Cluster counts in plotting order.
        ch:               Calinski–Harabasz medians.
        ch_low:           Lower bound (typically the 25th percentile).
        ch_high:          Upper bound (typically the 75th percentile).
        instability:      Hungarian-matched centroid-drift medians.
        instability_low:  Lower bound for instability.
        instability_high: Upper bound for instability.
        silhouette:       Stratified mean-silhouette medians.
        silhouette_low:   Lower bound for silhouette (over repeats).
        silhouette_high:  Upper bound for silhouette (over repeats).
        inertia:          W(k) medians.
        inertia_low:      Lower bound for inertia.
        inertia_high:     Upper bound for inertia.

    Returns:
        DataFrame with the median columns plus matching ``_low`` /
        ``_high`` columns.
    """
    n = len(k_values)

    def _col(seq):
        return list(seq) if seq is not None else [np.nan] * n

    return pd.DataFrame({
        "k":                     list(k_values),
        "calinski_harabasz":     _col(ch),
        "calinski_harabasz_low": _col(ch_low),
        "calinski_harabasz_high":_col(ch_high),
        "instability":           _col(instability),
        "instability_low":       _col(instability_low),
        "instability_high":      _col(instability_high),
        "silhouette":            _col(silhouette),
        "silhouette_low":        _col(silhouette_low),
        "silhouette_high":       _col(silhouette_high),
        "inertia":               _col(inertia),
        "inertia_low":           _col(inertia_low),
        "inertia_high":          _col(inertia_high),
    })
