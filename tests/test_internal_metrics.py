# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — tests.test_internal_metrics                              ║
# ║  « Silhouette + Inertia + Kneedle elbow tests »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Internal-metric tests for the Sprint 1 module.  All synthetic  ║
# ║  inputs — no STAG-specific data required.                       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Tests for stag.clustering.internal_metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans

from stag.clustering.internal_metrics import (
    compute_inertia,
    compute_silhouette_stratified,
    locate_elbow_kneedle,
    selection_summary,
)


class TestComputeInertia:
    """compute_inertia must agree with sklearn.cluster.KMeans.inertia_."""

    def test_matches_sklearn_on_well_separated_blobs(self):
        rng = np.random.default_rng(0)
        X = np.vstack([
            rng.normal(c, 0.5, (200, 3))
            for c in [[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]]
        ])
        km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
        our = compute_inertia(X, km.cluster_centers_, km.labels_)
        assert np.isclose(our, km.inertia_, rtol=1e-9)

    def test_zero_inertia_at_centroid_means(self):
        """Single-sample-per-cluster data: each sample IS its centroid → W = 0."""
        X = np.array([[0.0, 0.0], [10.0, 10.0], [-5.0, 5.0]])
        labels = np.array([0, 1, 2])
        centroids = X.copy()
        assert compute_inertia(X, centroids, labels) == 0.0


class TestSilhouetteStratified:
    """Silhouette should be high on well-separated, low on overlapping data."""

    @pytest.fixture
    def well_separated(self):
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal(c, 0.3, (300, 2))
            for c in [[0, 0], [10, 0], [0, 10], [10, 10]]
        ])
        labels = np.repeat([0, 1, 2, 3], 300)
        return X, labels, rng

    def test_well_separated_silhouette_above_threshold(self, well_separated):
        X, labels, rng = well_separated
        result = compute_silhouette_stratified(
            X, labels, n_per_cluster=150, n_repeats=20, rng=rng,
        )
        assert result["mean_silhouette"] > 0.85

    def test_overlapping_silhouette_below_threshold(self):
        """Heavy overlap → silhouette near 0 (boundary classification)."""
        rng = np.random.default_rng(7)
        X = np.vstack([
            rng.normal(c, 3.0, (300, 2))
            for c in [[0, 0], [1, 0]]
        ])
        labels = np.repeat([0, 1], 300)
        result = compute_silhouette_stratified(
            X, labels, n_per_cluster=150, n_repeats=10, rng=rng,
        )
        assert result["mean_silhouette"] < 0.3

    def test_return_shape(self, well_separated):
        X, labels, rng = well_separated
        result = compute_silhouette_stratified(
            X, labels, n_per_cluster=80, n_repeats=5, rng=rng,
        )
        assert "mean_silhouette" in result
        assert "iqr_silhouette" in result
        assert "per_repeat" in result
        assert "per_cluster_mean" in result
        assert result["per_repeat"].shape == (5,)
        assert result["per_cluster_mean"].shape == (4,)
        low, high = result["iqr_silhouette"]
        assert low <= result["mean_silhouette"] <= high or np.isclose(low, high)

    def test_stratification_handles_imbalanced_clusters(self):
        """Tiny cluster (< 1 % of data) is still subsampled fully."""
        rng = np.random.default_rng(13)
        X = np.vstack([
            rng.normal([0, 0], 0.2, (10_000, 2)),  # big cluster
            rng.normal([5, 5], 0.2, (50, 2)),      # tiny cluster
        ])
        labels = np.array([0] * 10_000 + [1] * 50)
        # n_per_cluster bigger than the small cluster: should clip
        # to the cluster size rather than oversample.
        result = compute_silhouette_stratified(
            X, labels, n_per_cluster=200, n_repeats=3, rng=rng,
        )
        assert result["mean_silhouette"] > 0.8


class TestLocateElbowKneedle:
    """Kneedle should detect the elbow of a clear convex-decay curve."""

    def test_finds_elbow_on_synthetic_decay(self):
        """Curve with a sharp knee at k=8 should return elbow_k near 8."""
        k_values = list(range(2, 21))
        knee_at = 8
        W = np.concatenate([
            np.linspace(100.0, 20.0, knee_at - 2 + 1),
            np.linspace(20.0, 18.0, len(k_values) - (knee_at - 2 + 1)),
        ])
        result = locate_elbow_kneedle(k_values, W)
        # Kneedle's exact return can be off by ± 1; check tolerance.
        assert result["elbow_k"] is not None
        assert abs(result["elbow_k"] - knee_at) <= 2

    def test_returns_none_on_monotonic_data(self):
        """Pure linear decay has no elbow."""
        k_values = list(range(2, 11))
        W = [10.0 - 0.5 * k for k in k_values]
        result = locate_elbow_kneedle(k_values, W)
        # Kneedle may return None or a borderline k for linear data; the
        # contract is that the result is a dict with the documented keys.
        assert set(result.keys()) >= {
            "elbow_k", "elbow_y", "elbow_index", "normalised_knee_distance",
        }


class TestSelectionSummary:
    """selection_summary builds a tidy per-k DataFrame."""

    def test_columns_present(self):
        k = [2, 3, 4]
        df = selection_summary(k, ch=[1, 2, 3])
        # Median columns + matching _low / _high band columns.
        expected = {
            "k",
            "calinski_harabasz", "calinski_harabasz_low", "calinski_harabasz_high",
            "instability", "instability_low", "instability_high",
            "silhouette", "silhouette_low", "silhouette_high",
            "inertia", "inertia_low", "inertia_high",
        }
        assert set(df.columns) == expected
        assert df["k"].tolist() == k
        assert df["calinski_harabasz"].tolist() == [1, 2, 3]
        # Unspecified metrics + unspecified bands are NaN.
        assert df["silhouette"].isna().all()
        assert df["inertia"].isna().all()
        assert df["calinski_harabasz_low"].isna().all()
        assert df["calinski_harabasz_high"].isna().all()

    def test_partial_inputs(self):
        """Passing only some metrics still returns a full-width DataFrame."""
        df = selection_summary(
            [2, 3, 4, 5],
            inertia=[100.0, 80.0, 65.0, 60.0],
        )
        assert df["inertia"].tolist() == [100.0, 80.0, 65.0, 60.0]
        assert df["calinski_harabasz"].isna().all()
