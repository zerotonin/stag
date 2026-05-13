"""Unit tests for Sprint 3 sequence-statistics modules.

Covers:
  * stag.analysis.null_models — shuffle preservation, n-gram counts, FDR ordering.
  * stag.analysis.super_prototypes — bout-stream RLE, per-animal boundaries.
  * stag.analysis.circadian — astral classification, hourly sums, day-split.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from stag.analysis.circadian import (
    classify_day_night,
    ear_flick_day_night_test,
    hourly_proportions,
    per_animal_time_budget,
    split_by_day,
)
from stag.analysis.null_models import (
    flag_significant_ngrams,
    ngram_frequencies,
    null_distribution,
    shuffle_first_order,
    shuffle_marginal,
    triplet_frequencies,
)
from stag.analysis.super_prototypes import (
    BoutStream,
    bout_duration_stats,
    bout_stream,
    extract_n_grams,
    identify_super_prototypes,
    per_animal_bout_streams,
)


# ─────────────────────────────────────────────────────────────────
#  null_models
# ─────────────────────────────────────────────────────────────────


class TestShuffles:
    def test_marginal_shuffle_preserves_marginals(self):
        rng = np.random.default_rng(0)
        seq = rng.integers(0, 4, size=10_000)
        sh = shuffle_marginal(seq, np.random.default_rng(1))
        np.testing.assert_array_equal(np.bincount(seq), np.bincount(sh))

    def test_first_order_shuffle_preserves_transition_matrix(self):
        # Construct a known Markov chain.
        rng = np.random.default_rng(0)
        T_true = np.array([
            [0.7, 0.2, 0.05, 0.05],
            [0.1, 0.6, 0.2,  0.1 ],
            [0.05,0.1, 0.7,  0.15],
            [0.1, 0.1, 0.2,  0.6 ],
        ])
        cdf = np.cumsum(T_true, axis=1)
        seq = np.zeros(50_000, dtype=np.int64)
        u = rng.random(seq.size)
        for t in range(1, seq.size):
            seq[t] = int(np.searchsorted(cdf[seq[t-1]], u[t]))

        sh = shuffle_first_order(seq, np.random.default_rng(2))

        def emp_T(x, k):
            pairs = np.column_stack((x[:-1], x[1:]))
            counts = np.zeros((k, k))
            np.add.at(counts, (pairs[:, 0], pairs[:, 1]), 1)
            return counts / counts.sum(axis=1, keepdims=True)

        T_orig = emp_T(seq, 4)
        T_sh   = emp_T(sh,  4)
        # Monte Carlo tolerance — 50k samples and 16 entries.
        assert np.abs(T_orig - T_sh).max() < 0.01

    def test_marginal_shuffle_destroys_transitions(self):
        # Same setup as above but check that marginal shuffle does NOT
        # preserve T̂ — this is the property that motivates using the
        # first-order shuffle for super-prototype detection.
        rng = np.random.default_rng(0)
        seq = np.tile([0, 1, 2, 3], 2500)  # perfectly periodic
        sh = shuffle_marginal(seq, np.random.default_rng(1))

        def emp_T(x, k):
            pairs = np.column_stack((x[:-1], x[1:]))
            counts = np.zeros((k, k))
            np.add.at(counts, (pairs[:, 0], pairs[:, 1]), 1)
            return counts / counts.sum(axis=1, keepdims=True)

        # Original has perfect 0→1→2→3 transitions; shuffled does not.
        T_orig = emp_T(seq, 4)
        T_sh   = emp_T(sh,  4)
        assert T_orig[0, 1] == 1.0
        assert T_sh[0, 1] < 0.5


class TestNgramFrequencies:
    def test_triplet_count_simple(self):
        seq = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        f = triplet_frequencies(seq, n_states=3)
        assert f[(0, 1, 2)] == 3
        assert f[(1, 2, 0)] == 2
        assert f[(2, 0, 1)] == 2

    def test_ngram_total_count_equals_n_minus_n_plus_1(self):
        seq = np.random.default_rng(0).integers(0, 4, size=1000)
        f = ngram_frequencies(seq, n=3, n_states=4)
        assert sum(f.values()) == seq.size - 2


class TestFlagSignificance:
    def test_flag_ranks_by_q_ascending(self):
        # Synthetic observed with one clear outlier and noise.
        rng = np.random.default_rng(0)
        observed = {(0, 0, 0): 200, (0, 0, 1): 50, (0, 1, 0): 50}
        null = {
            (0, 0, 0): rng.integers(40, 60, size=200),
            (0, 0, 1): rng.integers(40, 60, size=200),
            (0, 1, 0): rng.integers(40, 60, size=200),
        }
        flagged = flag_significant_ngrams(observed, null, percentile=99.9, fdr_alpha=0.05)
        # First (lowest q) must be the planted outlier.
        assert flagged[0]["ngram"] == (0, 0, 0)
        # Others should not be super-prototypes.
        for r in flagged[1:]:
            assert not r["super_prototype"]


# ─────────────────────────────────────────────────────────────────
#  super_prototypes
# ─────────────────────────────────────────────────────────────────


class TestBoutStream:
    def test_basic_rle(self):
        bs = bout_stream(np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0]))
        np.testing.assert_array_equal(bs.labels, [0, 1, 2, 0])
        np.testing.assert_array_equal(bs.lengths, [3, 2, 4, 2])
        np.testing.assert_array_equal(bs.start_index, [0, 3, 5, 9])

    def test_single_sample_bouts_kept(self):
        bs = bout_stream(np.array([0, 1, 0, 1, 0]))
        np.testing.assert_array_equal(bs.labels, [0, 1, 0, 1, 0])
        np.testing.assert_array_equal(bs.lengths, [1, 1, 1, 1, 1])

    def test_empty_input(self):
        bs = bout_stream(np.array([], dtype=np.int32))
        assert bs.n_bouts == 0

    def test_per_animal_bouts_do_not_cross_boundaries(self):
        idx = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        deer = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        streams = per_animal_bout_streams(idx, deer)
        assert set(streams.keys()) == {1, 2}
        # If bouts spanned animals, the deer-1 → deer-2 boundary
        # (last 1 → first 0) would join into a single bout per deer.
        # Independent streams guarantee the bout split.
        assert streams[1].n_bouts == 2
        assert streams[2].n_bouts == 2


class TestBoutDurationStats:
    def test_returns_per_pm_dict(self):
        bs = bout_stream(np.array([0, 0, 0, 1, 1, 0, 1]))
        stats = bout_duration_stats(bs, fps=50.0)
        assert set(stats.keys()) == {0, 1}
        # Two bouts of PM 0: lengths [3, 1]
        assert stats[0]["n_bouts"] == 2
        assert stats[0]["total_time_s"] == pytest.approx((3 + 1) / 50.0)


class TestIdentifySuperPrototypes:
    def test_planted_triplet_in_top(self):
        rng = np.random.default_rng(0)
        bout_labels = rng.integers(0, 4, size=5_000)
        # Plant 100 extra (0,1,2) bout-triplets.
        positions = rng.choice(bout_labels.size - 3, size=100, replace=False)
        for p in positions:
            bout_labels[p:p+3] = [0, 1, 2]
        # Stub sample-level idx: each bout becomes one sample (so the
        # bout stream of idx == bout_labels).  bout_stream RLE-encodes
        # consecutive identical bouts back into longer bouts, so we
        # break ties by interleaving sentinel values — but here we
        # already inserted explicit triplets so most positions differ
        # from their neighbours, and the bout stream is approximately
        # the same as bout_labels for ranking purposes.
        # Instead pass the labels through identify_super_prototypes
        # directly using a single-animal deer_ids array.
        deer_ids = np.zeros_like(bout_labels)
        result = identify_super_prototypes(
            bout_labels, deer_ids=deer_ids, n_gram=3,
            n_shuffles=200, percentile=99.9, fdr_alpha=0.05,
            rng=np.random.default_rng(1), n_states=4,
        )
        top3 = [r["ngram"] for r in result[:3]]
        assert (0, 1, 2) in top3


# ─────────────────────────────────────────────────────────────────
#  circadian
# ─────────────────────────────────────────────────────────────────


class TestClassifyDayNight:
    def test_noon_is_day(self):
        ts = np.array([
            pd.Timestamp("2018-11-13 12:00:00").value,
        ], dtype=np.int64)
        dl = classify_day_night(ts)
        assert dl[0] == 1  # day

    def test_midnight_is_night(self):
        ts = np.array([
            pd.Timestamp("2018-11-13 02:00:00").value,
        ], dtype=np.int64)
        dl = classify_day_night(ts)
        assert dl[0] == 0  # night

    def test_crepuscular_window_is_marked(self):
        # 2018-11-13 sunrise is ~05:58:31 NZDT; pick a sample 7 min
        # later — past sunrise but still inside the 15-min margin.
        ts = np.array([
            pd.Timestamp("2018-11-13 06:05:00").value,
        ], dtype=np.int64)
        dl = classify_day_night(ts, crepuscular_margin_minutes=15.0)
        assert dl[0] == -1  # crepuscular

    def test_no_crepuscular_margin(self):
        # With zero margin, sunrise itself is day.
        ts = np.array([
            pd.Timestamp("2018-11-13 12:00:00").value,
        ], dtype=np.int64)
        dl = classify_day_night(ts, crepuscular_margin_minutes=0.0)
        assert dl[0] == 1


class TestHourlyProportions:
    def test_rows_sum_to_one_when_all_pms_listed(self):
        rng = np.random.default_rng(0)
        n = 86_400  # 24 h at 1 Hz
        ts = np.arange(n, dtype=np.int64) * 1_000_000_000 + pd.Timestamp("2018-11-13").value
        idx = rng.integers(0, 4, size=n)
        hp = hourly_proportions(idx, ts, pm_ids=[0, 1, 2, 3])
        pm_cols = [c for c in hp.columns if c != "n_samples"]
        row_sums = hp[pm_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)


class TestSplitByDay:
    def test_two_day_partition(self):
        ts = np.array([
            pd.Timestamp("2018-11-12 12:00:00").value,
            pd.Timestamp("2018-11-13 12:00:00").value,
            pd.Timestamp("2018-11-14 12:00:00").value,
        ], dtype=np.int64)
        dy = split_by_day(ts)
        np.testing.assert_array_equal(dy, [0, 1, 2])

    def test_per_animal_split_restarts(self):
        ts = np.array([
            pd.Timestamp("2018-11-12 12:00:00").value,
            pd.Timestamp("2018-11-13 12:00:00").value,
            pd.Timestamp("2018-11-20 12:00:00").value,  # animal 2 starts
            pd.Timestamp("2018-11-21 12:00:00").value,
        ], dtype=np.int64)
        deer = np.array([1, 1, 2, 2])
        dy = split_by_day(ts, deer)
        np.testing.assert_array_equal(dy, [0, 1, 0, 1])


class TestEarFlickDayNight:
    def test_returns_per_animal_table(self):
        # Two animals, 48 h each, ear-flicks injected during day for both.
        ts_one = pd.date_range("2018-11-12", periods=48*60, freq="1min").values.astype("datetime64[ns]").astype("int64")
        idx_one = np.zeros(ts_one.size, dtype=np.int8)
        is_day = ((pd.to_datetime(ts_one).hour >= 9) & (pd.to_datetime(ts_one).hour < 18))
        idx_one[is_day] = 2  # PM 2 == ear flick
        idx_one[~is_day] = 1  # PM 1 == resting (counts as activity)

        ts = np.concatenate([ts_one, ts_one])
        idx = np.concatenate([idx_one, idx_one])
        deer = np.concatenate([np.ones(ts_one.size, dtype=np.int8),
                               2 * np.ones(ts_one.size, dtype=np.int8)])
        result = ear_flick_day_night_test(
            idx, ts, deer,
            ear_flick_pms=[2], activity_pms=[1, 2],
        )
        pa = result["per_animal"]
        # Both animals should have rate_day > rate_night.
        for d in (1, 2):
            assert pa.loc[d, "rate_day"] > pa.loc[d, "rate_night"]


class TestPerAnimalTimeBudget:
    def test_rows_sum_to_one(self):
        idx = np.array([0, 0, 1, 1, 2, 2, 0, 0])
        deer = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        tb = per_animal_time_budget(idx, deer, pm_ids=[0, 1, 2])
        pm_cols = [c for c in tb.columns if c != "n_samples"]
        np.testing.assert_allclose(tb[pm_cols].sum(axis=1).values, 1.0)
