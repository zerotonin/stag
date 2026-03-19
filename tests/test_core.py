"""
Test suite for the STAG pipeline.

These tests verify core functionality without requiring GPU hardware
or the full sensor dataset.
"""

import numpy as np
import pytest


class TestShrinkData:
    """Tests for the contiguous leave-out data reduction."""

    def test_no_reduction(self):
        """Zero deletion returns the original array unchanged."""
        from stag.clustering.kmeans import shrink_data

        data = np.arange(100).reshape(50, 2)
        result = shrink_data(data, reduction_percent=0, cut_position_percent=0)
        np.testing.assert_array_equal(result, data)

    def test_fifty_percent_reduction(self):
        """Fifty-percent deletion removes half the rows."""
        from stag.clustering.kmeans import shrink_data

        data = np.arange(200).reshape(100, 2)
        result = shrink_data(data, reduction_percent=50, cut_position_percent=0)
        assert result.shape[0] == 50

    def test_wraparound(self):
        """Cut wrapping past the end removes rows from both ends."""
        from stag.clustering.kmeans import shrink_data

        data = np.arange(200).reshape(100, 2)
        result = shrink_data(data, reduction_percent=25, cut_position_percent=90)
        # Wraparound deletes tail (rows 90-99 = 10) then head (5 more)
        assert result.shape[0] < 100
        assert result.shape[0] > 50  # sanity: not over-deleted


class TestSyncUtils:
    """Tests for synchronisation utility functions."""

    def test_correct_calibration_zero_mean(self):
        """Z-scored columns should have approximately zero mean."""
        import pandas as pd
        from stag.sync.utils import correct_calibration

        df = pd.DataFrame({"X": [1, 2, 3, 4, 5], "Y": [5, 4, 3, 2, 1], "Z": [2, 2, 2, 2, 2]})
        result = correct_calibration(df, cols=["X", "Y"])
        assert abs(result["X"].mean()) < 1e-10
        assert abs(result["Y"].mean()) < 1e-10

    def test_make_absolute(self):
        """Absolute transform removes negative values."""
        import pandas as pd
        from stag.sync.utils import make_absolute

        df = pd.DataFrame({"a": [-1, 2, -3], "b": [4, -5, 6]})
        result = make_absolute(df)
        assert (result >= 0).all().all()

    def test_sum_columns(self):
        """Row-wise sum matches expected values."""
        import pandas as pd
        from stag.sync.utils import sum_columns

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = sum_columns(df)
        np.testing.assert_array_equal(result.values, [5, 7, 9])


class TestLabelAnalyser:
    """Tests for the label analysis module using synthetic data."""

    @pytest.fixture
    def label_file(self, tmp_path):
        """Create a temporary .npy label file."""
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0])
        path = tmp_path / "labels.npy"
        np.save(path, labels)
        return str(path)

    def test_percentage(self, label_file):
        """Percentages should sum to 100."""
        from stag.analysis.label_analysis import LabelAnalyser

        la = LabelAnalyser(label_file, fps=1)
        pct = la.get_percentage()
        assert pytest.approx(pct.sum(), abs=1e-10) == 100.0

    def test_transitions_shape(self, label_file):
        """Transition matrix should be square with dimension = cen_num."""
        from stag.analysis.label_analysis import LabelAnalyser

        la = LabelAnalyser(label_file, fps=1)
        T = la.get_transitions()
        assert T.shape == (la.cen_num, la.cen_num)

    def test_transitions_sum(self, label_file):
        """Total transitions should equal len(labels) - 1."""
        from stag.analysis.label_analysis import LabelAnalyser

        la = LabelAnalyser(label_file, fps=1)
        T = la.get_transitions()
        assert T.sum() == la.label_num - 1

    def test_mean_durations(self, label_file):
        """Mean durations should be positive for all labels."""
        from stag.analysis.label_analysis import LabelAnalyser

        la = LabelAnalyser(label_file, fps=1)
        durations = la.get_mean_durations()
        for mean, sem in durations:
            assert mean > 0

    def test_save_json(self, label_file, tmp_path):
        """Full pipeline should produce a valid JSON output."""
        import json
        from stag.analysis.label_analysis import LabelAnalyser

        la = LabelAnalyser(label_file, fps=1)
        out = str(tmp_path / "results.json")
        la.main(cutoff=1, save_path=out)

        with open(out) as f:
            data = json.load(f)
        assert "centroids" in data
        assert "transition_matrix" in data


class TestFilenameGenerator:
    """Tests for the clustering filename generator."""

    def test_generates_three_keys(self, tmp_path):
        """Should return centroids, labels, and meta paths."""
        from stag.clustering.kmeans import generate_filename

        result = generate_filename(str(tmp_path), "test", 8, 0, 0)
        assert set(result.keys()) == {"centroids", "labels", "meta"}
        assert result["meta"].endswith(".json")
        assert result["centroids"].endswith(".npy")


class TestGPSTortuosity:
    """Tests for tortuosity calculation."""

    def test_straight_line_tortuosity(self):
        """A straight trajectory should yield tortuosity near 1.0."""
        from stag.gps.tortuosity import calculate_tortuosity_and_speed

        lat = np.array([0.0, 0.001, 0.002, 0.003, 0.004])
        lon = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        result = calculate_tortuosity_and_speed(lat, lon, fps=0.5)
        tort = result["tortuosity"]
        # Interior values should be close to 1.0
        for val in tort[1:-1]:
            assert pytest.approx(val, abs=0.05) == 1.0
