# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — analysis.stability_null                                  ║
# ║  « cluster-stability d-ceiling via a structureless surrogate »   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Implements the R3 cluster-stability null model: draws a         ║
# ║  uniform reference distribution on the MaxAbs box [-1, 1]^d at   ║
# ║  the same n as the real feature matrix, then quantifies the      ║
# ║  Hungarian-matched centroid drift across an ensemble of k-means  ║
# ║  fits on that surrogate.  The resulting per-k drift curve is     ║
# ║  the d-ceiling that anchors Panel B of the four-reduction        ║
# ║  internal-metrics figure.                                        ║
# ║                                                                  ║
# ║  This is the reference-distribution null in the Tibshirani /     ║
# ║  Walther / Hastie (2001) gap-statistic sense, applied to the     ║
# ║  same Hungarian-matched centroid drift metric the real-data      ║
# ║  panel uses.  It is deliberately distinct from the label-        ║
# ║  shuffle nulls in :mod:`stag.analysis.null_models`, which test   ║
# ║  super-prototype claims rather than centroid reproducibility.    ║
# ║                                                                  ║
# ║  References                                                      ║
# ║   - Tibshirani, Walther & Hastie (2001) Estimating the number    ║
# ║     of clusters in a data set via the gap statistic. JRSS B 63.  ║
# ║   - Ben-Hur, Elisseeff & Guyon (2002) A stability based method   ║
# ║     for discovering structure in clustered data. PSB 7: 6-17.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Cluster-stability null model — uniform MaxAbs-box surrogate."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def build_uniform_surrogate(
    n: int,
    d: int,
    seed: int,
    dtype: type = np.float32,
) -> np.ndarray:
    """Draw a structureless reference on the MaxAbs box ``[-1, 1]^d``.

    Args:
        n:     Sample count, matched to the real feature matrix.
        d:     Feature count (six accelerometer columns for the deer-2024
               clustering).
        seed:  PRNG seed for reproducible surrogate draws.
        dtype: Output dtype.  float32 halves disk + I/O cost versus the
               float64 real input without changing cuML k-means behaviour.

    Returns:
        Array of shape ``(n, d)`` of i.i.d. ``Uniform(-1, 1)`` draws.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, d)).astype(dtype)


def hungarian_centroid_drift(
    centroids_list: list[np.ndarray],
    ch_scores: list[float] | None = None,
    unscale: np.ndarray | None = None,
) -> np.ndarray:
    """Hungarian-matched mean centroid distance to the best-CH reference.

    Mirrors the within-``(delSize, k)`` drift computation used by
    :mod:`scripts.replot_internal_metrics_4reductions`: one fit is the
    basin reference (highest Calinski-Harabasz score when ``ch_scores``
    is provided, otherwise fit index 0), and every other fit is matched
    against it via the linear-sum assignment on the centroid-distance
    matrix.  The per-fit return value is the mean Euclidean distance
    between matched centroids.

    Args:
        centroids_list: List of ``(k, d)`` arrays, one per k-means fit.
        ch_scores:      Optional Calinski-Harabasz scores aligned with
                        ``centroids_list``.  When provided, the
                        highest-CH fit becomes the reference.
        unscale:        Optional per-column multiplier of shape ``(d,)``
                        applied to every centroid before the distance
                        calc.  Use this to lift centroids out of MaxAbs
                        scale ``[-1, 1]`` back into the original
                        per-feature units (e.g. g-force for the deer
                        accelerometer columns).  Distances are uniformly
                        scaled when every divisor is identical, so the
                        null-vs-real ratio is preserved; the conversion
                        only changes the y-axis units, not the
                        scientific finding.  ``None`` keeps the MaxAbs
                        convention.

    Returns:
        Length-``len(centroids_list)`` array of per-fit drift values.
        The reference fit's entry is ``NaN``.
    """
    n_fits = len(centroids_list)
    drift = np.full(n_fits, np.nan)
    if n_fits < 2:
        return drift

    if ch_scores is None:
        ref_idx = 0
    else:
        ref_idx = int(np.nanargmax(np.asarray(ch_scores, dtype=np.float64)))

    scale = (
        None if unscale is None
        else np.asarray(unscale, dtype=np.float32).reshape(1, -1)
    )

    def _prep(c: np.ndarray) -> np.ndarray:
        arr = np.asarray(c, dtype=np.float32)
        if scale is not None:
            arr = arr * scale
        return arr

    ref_c = _prep(centroids_list[ref_idx])
    for i in range(n_fits):
        if i == ref_idx:
            continue
        other_c = _prep(centroids_list[i])
        dist_matrix = np.linalg.norm(
            ref_c[:, None, :] - other_c[None, :, :], axis=2,
        )
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        drift[i] = float(dist_matrix[row_ind, col_ind].mean())
    return drift
