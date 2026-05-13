# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — analysis.null_models                                     ║
# ║  « shuffle nulls + n-gram frequency significance »               ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Sprint 3 / R3 Q5: the manuscript's "super-prototype" claim is   ║
# ║  that certain n-grams (triplets like PM2→PM5→PM4) appear more    ║
# ║  often than would be predicted from the pairwise transition      ║
# ║  matrix alone.  This module operationalises that claim with      ║
# ║  two shuffle nulls and a frequency-significance helper.          ║
# ║                                                                  ║
# ║  shuffle_marginal — preserves P(state).  Sanity-comparator       ║
# ║    only; over-rejects because it destroys transitions we         ║
# ║    already accept.                                               ║
# ║                                                                  ║
# ║  shuffle_first_order — preserves P(state) AND the empirical      ║
# ║    transition matrix T[i,j]=P(j|i).  Destroys only higher-order  ║
# ║    structure.  This is the principled null for the super-        ║
# ║    prototype claim: an n-gram beats this null only when its      ║
# ║    n-th step requires more than one-step memory of context.      ║
# ║                                                                  ║
# ║  Implementation note — sampling a first-order chain of length    ║
# ║  ~204 M states from a 26×26 transition matrix takes a few        ║
# ║  seconds with the cumulative-CDF trick; 1000 shuffles is         ║
# ║  realistic but unnecessary on the full sequence.  We provide     ║
# ║  ``null_distribution`` which accepts any IDX (full sequence,     ║
# ║  per-animal slice, or subsampled).                               ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Shuffle nulls and n-gram frequency significance."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
#  Shuffle nulls
# ─────────────────────────────────────────────────────────────────


def shuffle_marginal(idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of ``idx``.

    Preserves ``P(state)``; destroys all temporal structure including
    the transition matrix.  Cheap and useful as a sanity comparator
    — but for the super-prototype claim use :func:`shuffle_first_order`.
    """
    out = idx.copy()
    rng.shuffle(out)
    return out


def shuffle_first_order(
    idx: np.ndarray, rng: np.random.Generator,
    n_states: int | None = None,
) -> np.ndarray:
    """Sample a Markov chain matching ``idx``'s empirical transitions.

    Estimates the empirical transition matrix ``T̂`` and the empirical
    starting distribution from ``idx``, then samples a new chain of
    the same length.  Preserves ``P(state)`` *and* ``P(j|i)``;
    destroys only higher-order memory.

    Args:
        idx:       Per-sample state IDs in observed order, ``(n,)``.
        rng:       NumPy generator (use one per replicate for
                   independent shuffles).
        n_states:  Total number of states.  Inferred from ``idx`` if
                   ``None``.  Specify when ``idx`` is a slice that
                   may not contain every state.

    Returns:
        Synthetic chain of shape ``(n,)``, dtype matching ``idx``.
    """
    idx = np.asarray(idx)
    n = idx.size
    if n_states is None:
        n_states = int(idx.max()) + 1

    # Empirical transition matrix via 2D histogram of consecutive pairs.
    pairs = np.column_stack((idx[:-1], idx[1:]))
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    np.add.at(counts, (pairs[:, 0], pairs[:, 1]), 1)

    # Row-normalise to P(next | current).  Rows with zero outgoing
    # transitions (state never observed mid-sequence) get a uniform
    # row so the sampler does not stall on them — the row will only
    # be selected if its state appears as the starting sample.
    row_sums = counts.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    T = counts / safe
    zero_rows = (row_sums.ravel() == 0)
    if zero_rows.any():
        T[zero_rows] = 1.0 / n_states

    # Cumulative CDF per row — one searchsorted per emitted step.
    cdf = np.cumsum(T, axis=1)

    # Starting distribution: empirical state proportions.
    p0 = np.bincount(idx, minlength=n_states).astype(np.float64)
    p0 /= p0.sum()
    p0_cdf = np.cumsum(p0)

    out = np.empty(n, dtype=idx.dtype)
    u = rng.random(n)
    out[0] = int(np.searchsorted(p0_cdf, u[0]))
    for t in range(1, n):
        out[t] = int(np.searchsorted(cdf[out[t - 1]], u[t]))
    return out


# ─────────────────────────────────────────────────────────────────
#  n-gram frequency
# ─────────────────────────────────────────────────────────────────


def ngram_frequencies(
    idx: np.ndarray, n: int = 3, n_states: int | None = None,
) -> dict[tuple[int, ...], int]:
    """Count contiguous n-grams in ``idx``.

    Args:
        idx:       Per-sample state IDs.
        n:         n-gram length (default 3 = triplets).
        n_states:  Optional cap; ngrams containing states ≥ ``n_states``
                   are dropped.  Useful for restricting to the canonical
                   k=8 alphabet when ``idx`` has any noise.

    Returns:
        Dict mapping ``(s0, s1, …, s_{n-1})`` to its count in ``idx``.
        Sliding window with stride 1.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if n_states is not None:
        valid = idx < n_states
        # We need n consecutive valid positions to form a window.
        # Cheap and correct: build the n-tuple matrix first, then mask
        # rows where any element is out-of-range.
        pass

    if idx.size < n:
        return {}

    # Pack n consecutive states into a single int64 (works for n ≤ 7
    # at n_states ≤ 256; deer manuscript has n=3, n_states=8).
    if n_states is None:
        n_states = int(idx.max()) + 1
    multiplier = n_states ** np.arange(n, dtype=np.int64)
    cols = np.stack([idx[i:idx.size - (n - 1) + i] for i in range(n)], axis=1)
    if n_states is not None and (cols >= n_states).any():
        good = (cols < n_states).all(axis=1)
        cols = cols[good]
    keys = cols @ multiplier
    uniq, counts = np.unique(keys, return_counts=True)

    out: dict[tuple[int, ...], int] = {}
    for key, c in zip(uniq.tolist(), counts.tolist()):
        tup = tuple(int((key // multiplier[i]) % n_states) for i in range(n))
        out[tup] = int(c)
    return out


def triplet_frequencies(
    idx: np.ndarray, n_states: int | None = None,
) -> dict[tuple[int, int, int], int]:
    """Convenience wrapper: 3-gram frequencies."""
    return ngram_frequencies(idx, n=3, n_states=n_states)  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────
#  Null distribution and significance
# ─────────────────────────────────────────────────────────────────


def null_distribution(
    idx: np.ndarray,
    n: int = 3,
    n_shuffles: int = 1000,
    null_kind: str = "first_order",
    rng: np.random.Generator | None = None,
    n_states: int | None = None,
    desc: str | None = None,
) -> dict[tuple[int, ...], np.ndarray]:
    """Build the null frequency distribution for every observed n-gram.

    Runs ``n_shuffles`` shuffles of the requested null and records
    each candidate n-gram's count per shuffle.  The result is a dict
    ``{ngram: shuffle_counts}`` aligned in iteration order; the
    union of n-grams seen in any shuffle plus the empirical n-grams
    is what we test in :func:`flag_significant_ngrams`.

    Args:
        idx:        Observed state sequence.
        n:          n-gram length.
        n_shuffles: Number of shuffle replicates.
        null_kind:  ``"first_order"`` or ``"marginal"``.
        rng:        Seeded ``numpy.random.Generator``.  ``None`` → fresh.
        n_states:   See :func:`shuffle_first_order`.
        desc:       Optional tqdm description.

    Returns:
        Dict from ngram tuple to a ``(n_shuffles,)`` int array of
        per-shuffle counts.
    """
    if rng is None:
        rng = np.random.default_rng()
    if null_kind not in ("first_order", "marginal"):
        raise ValueError(f"null_kind={null_kind!r} — expected 'first_order' or 'marginal'")
    shuffle_fn = (
        shuffle_first_order if null_kind == "first_order" else
        lambda x, r: shuffle_marginal(x, r)
    )
    if n_states is None:
        n_states = int(np.asarray(idx).max()) + 1

    # First pass: union of n-grams ever seen across shuffles + observed.
    # We cannot know the universe up front, so accumulate.
    accum: dict[tuple[int, ...], list[int]] = {}

    observed = ngram_frequencies(idx, n=n, n_states=n_states)
    for ng in observed:
        accum.setdefault(ng, [])

    iterator = range(n_shuffles)
    if desc:
        iterator = tqdm(iterator, desc=desc)
    for s in iterator:
        if null_kind == "first_order":
            sh = shuffle_first_order(idx, rng, n_states=n_states)
        else:
            sh = shuffle_marginal(idx, rng)
        freqs = ngram_frequencies(sh, n=n, n_states=n_states)
        for ng in accum:
            accum[ng].append(freqs.get(ng, 0))
        # ngrams that appear only in this shuffle but never in observed
        # need a column of zeros for prior shuffles.
        for ng in freqs:
            if ng not in accum:
                accum[ng] = [0] * (s + 1)
                accum[ng][-1] = freqs[ng]

    return {ng: np.asarray(v, dtype=np.int64) for ng, v in accum.items()}


def flag_significant_ngrams(
    observed: dict[tuple[int, ...], int],
    null:     dict[tuple[int, ...], np.ndarray],
    percentile: float = 99.9,
    fdr_alpha:  float = 0.05,
) -> list[dict]:
    """Identify n-grams whose observed count exceeds the null tail.

    For each n-gram seen empirically:
      * percentile cutoff — observed > ``np.quantile(null, percentile/100)``
      * empirical p-value — ``(1 + Σ𝟙[null ≥ observed]) / (1 + n_shuffles)``
      * BH-FDR q-value — Benjamini–Hochberg across all tested n-grams.

    A "super-prototype" call requires *both* the percentile flag and
    the FDR flag (Boolean AND).  Returns a list of dicts ordered by
    ascending q (most significant first).

    Args:
        observed:   ``ngram → observed_count``.
        null:       ``ngram → (n_shuffles,) array of null counts``.
        percentile: Percentile cutoff for the percentile flag.
        fdr_alpha:  Family-wise FDR control level.

    Returns:
        List of dicts with keys: ``ngram``, ``observed``, ``null_median``,
        ``null_q975``, ``null_qP``, ``percentile_flag``, ``p_empirical``,
        ``q_bh``, ``fdr_flag``, ``super_prototype`` (= percentile_flag &
        fdr_flag).
    """
    if not observed:
        return []
    if not null:
        raise ValueError("Empty null distribution.")

    items = list(observed.items())
    ngrams = [ng for ng, _ in items]
    n_per_test = len(items)

    p_vals = np.empty(n_per_test)
    rows = []
    for i, (ng, obs) in enumerate(items):
        null_counts = null.get(ng, np.zeros(1, dtype=np.int64))
        n_sh = null_counts.size
        p = (1 + int((null_counts >= obs).sum())) / (1 + n_sh)
        p_vals[i] = p
        rows.append({
            "ngram":             ng,
            "observed":          int(obs),
            "null_median":       float(np.median(null_counts)),
            "null_q975":         float(np.quantile(null_counts, 0.975)) if n_sh > 1 else float("nan"),
            "null_qP":           float(np.quantile(null_counts, percentile / 100.0)) if n_sh > 1 else float("nan"),
            "percentile_flag":   bool(obs > float(np.quantile(null_counts, percentile / 100.0))) if n_sh > 1 else False,
            "p_empirical":       float(p),
        })

    # Benjamini-Hochberg.
    order = np.argsort(p_vals)
    ranked = p_vals[order] * n_per_test / np.arange(1, n_per_test + 1)
    # enforce monotonicity from the largest q downwards
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_vals = np.empty_like(q_sorted)
    q_vals[order] = q_sorted

    for i, row in enumerate(rows):
        row["q_bh"]            = float(q_vals[i])
        row["fdr_flag"]        = bool(q_vals[i] < fdr_alpha)
        row["super_prototype"] = row["percentile_flag"] and row["fdr_flag"]

    rows.sort(key=lambda r: r["q_bh"])
    return rows


# ─────────────────────────────────────────────────────────────────
#  Convenience: top-N supplementary table
# ─────────────────────────────────────────────────────────────────


def top_n_super_prototypes(
    idx: np.ndarray,
    n_gram: int = 3,
    n_shuffles: int = 1000,
    percentile: float = 99.9,
    fdr_alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    n_states: int | None = None,
    desc: str | None = None,
) -> list[dict]:
    """One-shot driver: empirical n-grams → null → significance.

    Equivalent to::

        observed = ngram_frequencies(idx, n_gram, n_states)
        null     = null_distribution(idx, n_gram, n_shuffles, "first_order", rng, n_states)
        return flag_significant_ngrams(observed, null, percentile, fdr_alpha)
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_states is None:
        n_states = int(np.asarray(idx).max()) + 1
    observed = ngram_frequencies(idx, n=n_gram, n_states=n_states)
    null = null_distribution(
        idx, n=n_gram, n_shuffles=n_shuffles, null_kind="first_order",
        rng=rng, n_states=n_states, desc=desc,
    )
    return flag_significant_ngrams(observed, null, percentile=percentile, fdr_alpha=fdr_alpha)
