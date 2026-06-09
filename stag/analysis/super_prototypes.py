# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — analysis.super_prototypes                                ║
# ║  « bout-level n-grams of cluster labels »                        ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The manuscript's "super-prototypes" — `PM2 → PM5 → PM4`         ║
# ║  composite ear-flicks, `PM6 ↔ PM7` grazing-stepping — are        ║
# ║  *bout-level* triplets, not three-consecutive-sample triplets.  ║
# ║  Running the null-model triplet test directly on the raw 50 Hz  ║
# ║  sample sequence would produce a trivial answer: every common   ║
# ║  triplet would be `(PM0, PM0, PM0)`-style dwell sequences that  ║
# ║  reflect the sticky single-state runs already explained by the  ║
# ║  pairwise transition matrix.                                    ║
# ║                                                                  ║
# ║  This module converts a per-sample IDX into a per-bout label    ║
# ║  stream (run-length encoded), then runs the n-gram null model   ║
# ║  on the bout stream.  A "super-prototype" call therefore        ║
# ║  requires that the *sequence of three different (or repeating)  ║
# ║  bouts* appears more often than a first-order Markov model of   ║
# ║  bout transitions can explain.                                  ║
# ║                                                                  ║
# ║  Both per-animal and pooled-across-animals analyses are         ║
# ║  supported: the bout stream is built per animal (bouts cannot   ║
# ║  cross deer boundaries) and either tested separately or         ║
# ║  concatenated with sentinel breaks for a global test.           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Bout-level n-grams: run-length encode IDX, then super-prototype detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stag.analysis.null_models import (
    flag_significant_ngrams,
    ngram_frequencies,
    null_distribution,
)


@dataclass(frozen=True)
class BoutStream:
    """Run-length-encoded label sequence.

    Attributes:
        labels:       Per-bout cluster label, shape ``(n_bouts,)``.
        lengths:      Per-bout duration in samples, shape ``(n_bouts,)``.
        start_index:  Per-bout starting sample index in the original
                      IDX, shape ``(n_bouts,)``.
    """

    labels: np.ndarray
    lengths: np.ndarray
    start_index: np.ndarray

    @property
    def n_bouts(self) -> int:
        """Number of bouts in the stream (equals ``labels.size``)."""
        return int(self.labels.size)


# ─────────────────────────────────────────────────────────────────
#  Run-length encoding
# ─────────────────────────────────────────────────────────────────


def bout_stream(idx: np.ndarray) -> BoutStream:
    """Run-length-encode ``idx`` into bouts.

    A bout is a maximal contiguous run of the same label.  Length-1
    bouts (single-sample flickers between two longer dwells) are
    kept — filtering them is an interpretation-layer decision, not
    something this module should hide.

    Args:
        idx: Per-sample label sequence, ``(n,)`` integer.

    Returns:
        :class:`BoutStream` with three arrays of length ``n_bouts``.
    """
    idx = np.asarray(idx)
    if idx.size == 0:
        empty = np.zeros(0, dtype=idx.dtype)
        return BoutStream(labels=empty, lengths=empty.astype(np.int64),
                          start_index=empty.astype(np.int64))

    change = np.empty(idx.size, dtype=bool)
    change[0] = True
    change[1:] = idx[1:] != idx[:-1]
    start_index = np.flatnonzero(change)
    labels = idx[start_index]
    lengths = np.diff(np.append(start_index, idx.size))
    return BoutStream(
        labels=labels.astype(idx.dtype),
        lengths=lengths.astype(np.int64),
        start_index=start_index.astype(np.int64),
    )


def per_animal_bout_streams(
    idx: np.ndarray, deer_ids: np.ndarray,
) -> dict[int, BoutStream]:
    """Build one :class:`BoutStream` per unique ``deer_id``.

    Bouts are computed *within* an animal's recording and never
    cross between animals — important when the saved per-sample
    arrays concatenate multiple animals' tracks.
    """
    idx = np.asarray(idx)
    deer_ids = np.asarray(deer_ids)
    if idx.shape != deer_ids.shape:
        raise ValueError(
            f"idx shape {idx.shape} != deer_ids shape {deer_ids.shape}",
        )
    out: dict[int, BoutStream] = {}
    for d in np.unique(deer_ids):
        mask = deer_ids == d
        if not mask.any():
            continue
        out[int(d)] = bout_stream(idx[mask])
    return out


# ─────────────────────────────────────────────────────────────────
#  Bout-level n-gram extraction
# ─────────────────────────────────────────────────────────────────


def extract_n_grams(
    bouts: BoutStream | np.ndarray, n: int = 3, n_states: int | None = None,
) -> dict[tuple[int, ...], int]:
    """Count bout-level n-grams.

    Accepts either a :class:`BoutStream` (uses its ``labels``) or a
    1-D array of bout labels.  Sliding-window stride 1 on the bout
    sequence — so a triplet ``(a, b, c)`` is one bout of ``a``
    followed by one bout of ``b`` followed by one bout of ``c``.
    """
    label_seq = bouts.labels if isinstance(bouts, BoutStream) else np.asarray(bouts)
    return ngram_frequencies(label_seq, n=n, n_states=n_states)


def extract_triplets(
    bouts: BoutStream | np.ndarray, n_states: int | None = None,
) -> dict[tuple[int, int, int], int]:
    """Convenience wrapper for ``n = 3``."""
    return extract_n_grams(bouts, n=3, n_states=n_states)  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────
#  Significance and ranking
# ─────────────────────────────────────────────────────────────────


def identify_super_prototypes(
    idx: np.ndarray,
    deer_ids: np.ndarray | None = None,
    n_gram: int = 3,
    n_shuffles: int = 1000,
    percentile: float = 99.9,
    fdr_alpha: float = 0.05,
    null_kind: str = "first_order",
    rng: np.random.Generator | None = None,
    n_states: int | None = None,
    desc: str | None = None,
) -> list[dict]:
    """Identify bout-level n-grams that beat the first-order Markov null.

    Pipeline:
      1. Build the bout stream (per-animal if ``deer_ids`` given,
         single global stream otherwise).
      2. Concatenate per-animal bout-label streams into a single
         sequence for the null test.  Sentinel breaks are not
         inserted — the null is applied to the *concatenated bout
         sequence*, which preserves the empirical bout-to-bout
         transition matrix that is the whole point of the first-
         order null.  Inter-animal "transitions" are a tiny minority
         (n_animals − 1 of ~10 M bout-pairs in the cohort) and do
         not bias the percentile cutoff.
      3. Run :func:`null_distribution` and
         :func:`flag_significant_ngrams` on the bout sequence.

    Args:
        idx:         Per-sample cluster IDs, ``(n,)``.
        deer_ids:    Per-sample deer_id, ``(n,)`` — required when
                     ``idx`` spans multiple animals so bouts do not
                     cross animal boundaries.
        n_gram:      n-gram length (default 3 = triplets).
        n_shuffles:  Replicates for the null (manuscript: 1000).
        percentile:  Percentile cutoff for the percentile flag.
        fdr_alpha:   BH-FDR control level.
        null_kind:   ``"first_order"`` (default) or ``"marginal"``.
        rng:         Seeded ``numpy.random.Generator``.  None → fresh.
        n_states:    Optional state-count override.
        desc:        Optional tqdm description for the null pass.

    Returns:
        List of dicts as in :func:`flag_significant_ngrams`, sorted
        by ascending q-value (most significant first).  Each dict
        has the ``super_prototype`` boolean.
    """
    if rng is None:
        rng = np.random.default_rng()

    if deer_ids is None:
        streams = {0: bout_stream(np.asarray(idx))}
    else:
        streams = per_animal_bout_streams(idx, deer_ids)

    bout_labels = np.concatenate([s.labels for s in streams.values()])
    if n_states is None:
        n_states = int(bout_labels.max()) + 1

    observed = ngram_frequencies(bout_labels, n=n_gram, n_states=n_states)
    null = null_distribution(
        bout_labels, n=n_gram, n_shuffles=n_shuffles,
        null_kind=null_kind, rng=rng, n_states=n_states, desc=desc,
    )
    return flag_significant_ngrams(
        observed, null,
        percentile=percentile, fdr_alpha=fdr_alpha,
    )


# ─────────────────────────────────────────────────────────────────
#  Per-animal aggregation
# ─────────────────────────────────────────────────────────────────


def bout_duration_stats(
    bouts: BoutStream, fps: float,
) -> dict[int, dict[str, float]]:
    """Per-PM bout-duration summary (median, IQR, count) in seconds.

    Operates on a SINGLE :class:`BoutStream` and pools all of its bouts.
    For across-animal summaries use :func:`per_animal_pm_duration_stats`
    followed by :func:`aggregate_durations_across_animals` — pooling
    across animals at this level pseudo-replicates (the animal with
    most bouts dominates the moments and SEMs).
    """
    out: dict[int, dict[str, float]] = {}
    for pm in np.unique(bouts.labels):
        mask = bouts.labels == pm
        dur_s = bouts.lengths[mask] / fps
        if dur_s.size == 0:
            continue
        out[int(pm)] = {
            "n_bouts":       int(dur_s.size),
            "median_s":      float(np.median(dur_s)),
            "q25_s":         float(np.quantile(dur_s, 0.25)),
            "q75_s":         float(np.quantile(dur_s, 0.75)),
            "mean_s":        float(np.mean(dur_s)),
            "max_s":         float(dur_s.max()),
            "total_time_s":  float(dur_s.sum()),
        }
    return out


# ─────────────────────────────────────────────────────────────────
#  Per-animal-first duration aggregation
# ─────────────────────────────────────────────────────────────────


def per_animal_pm_duration_stats(
    streams: dict[int, "BoutStream"],
    fps: float,
) -> pd.DataFrame:
    """Per-(animal, PM) bout-duration summary.

    The animal is the unit of observation here.  This function emits one
    row per non-empty (``deer_id``, ``pm``) cell; callers feed the result
    to :func:`aggregate_durations_across_animals` to get cohort-level
    summaries without pseudo-replicating across the millions of pooled
    bouts.

    Args:
        streams: ``deer_id → BoutStream`` mapping (typically the output of
                 :func:`per_animal_bout_streams`).
        fps:     Sample rate in Hz so durations come out in seconds.

    Returns:
        DataFrame with columns ``deer_id``, ``pm``, ``n_bouts``,
        ``mean_s``, ``median_s``, ``q25_s``, ``q75_s``.
    """
    rows: list[dict[str, float | int]] = []
    for deer_id, stream in streams.items():
        for pm in np.unique(stream.labels):
            mask = stream.labels == pm
            dur_s = stream.lengths[mask] / fps
            if dur_s.size == 0:
                continue
            rows.append({
                "deer_id":  int(deer_id),
                "pm":       int(pm),
                "n_bouts":  int(dur_s.size),
                "mean_s":   float(np.mean(dur_s)),
                "median_s": float(np.median(dur_s)),
                "q25_s":    float(np.quantile(dur_s, 0.25)),
                "q75_s":    float(np.quantile(dur_s, 0.75)),
            })
    return pd.DataFrame(rows)


def aggregate_durations_across_animals(
    per_animal: pd.DataFrame,
) -> pd.DataFrame:
    """Cohort-level summary of per-(animal, PM) duration values.

    Two complementary aggregations per PM:

    - mean-of-means + SEM (matches the manuscript's existing duration
      style; SEM is across animals, *n = n_animals*, not across bouts).
    - median-of-medians + IQR (robust to the heavy right-skew of bout
      durations; IQR is taken across the per-animal medians).

    Args:
        per_animal: Output of :func:`per_animal_pm_duration_stats`.

    Returns:
        DataFrame with columns ``pm``, ``n_animals``,
        ``mean_of_means_s``, ``sem_s``, ``median_of_medians_s``,
        ``q25_of_medians_s``, ``q75_of_medians_s``,
        ``total_bouts``.
    """
    if per_animal.empty:
        return pd.DataFrame(
            columns=[
                "pm", "n_animals",
                "mean_of_means_s", "sem_s",
                "median_of_medians_s",
                "q25_of_medians_s", "q75_of_medians_s",
                "total_bouts",
            ],
        )
    rows: list[dict[str, float | int]] = []
    for pm, sub in per_animal.groupby("pm"):
        means = sub["mean_s"].to_numpy()
        medians = sub["median_s"].to_numpy()
        n_animals = int(means.size)
        # ddof=1 → sample standard deviation; SEM uses n_animals.
        sem = float(np.std(means, ddof=1) / np.sqrt(n_animals)) if n_animals > 1 else float("nan")
        rows.append({
            "pm":                    int(pm),
            "n_animals":             n_animals,
            "mean_of_means_s":       float(np.mean(means)),
            "sem_s":                 sem,
            "median_of_medians_s":   float(np.median(medians)),
            "q25_of_medians_s":      float(np.quantile(medians, 0.25)),
            "q75_of_medians_s":      float(np.quantile(medians, 0.75)),
            "total_bouts":           int(sub["n_bouts"].sum()),
        })
    return pd.DataFrame(rows).sort_values("pm").reset_index(drop=True)
