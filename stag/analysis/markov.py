# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — analysis.markov                                          ║
# ║  « first-order Markov transition structure »                     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Transition-matrix construction, marginal base rates, and        ║
# ║  above / below-baseline flagging for the prototype label         ║
# ║  sequence.  These functions formalise the per-target-PM          ║
# ║  marginal comparison that drives the up/down triangles on        ║
# ║  Figure 4A and supplies the inputs to the super-prototype        ║
# ║  triplet analysis (Sprint 3).                                    ║
# ║                                                                  ║
# ║  All functions are stateless — they operate on integer label     ║
# ║  arrays.  See stag.analysis.label_analysis.LabelAnalyser for     ║
# ║  the object-oriented entry point that wraps these helpers.       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""First-order Markov transition structure for prototype label sequences."""

from __future__ import annotations

import numpy as np

from stag.constants import FPS


def build_transition_matrix(
    IDX: np.ndarray,
    n_states: int | None = None,
    smoothing: float = 0.0,
    treat_as_single_chain: bool = True,
) -> np.ndarray:
    """Count first-order transitions in a label sequence.

    Args:
        IDX:                   Integer label array of shape ``(n_samples,)``.
        n_states:              Number of distinct labels; inferred from
                               ``IDX.max() + 1`` when None.
        smoothing:             Pseudo-count added to every cell before
                               returning (Laplace smoothing).  Default 0.0
                               (raw counts, matching the manuscript).
        treat_as_single_chain: When True (default), the array is treated
                               as one continuous 48-h chain.  Reserved for
                               future segmented-chain handling.

    Returns:
        ``(n_states, n_states)`` matrix where entry ``(i, j)`` counts
        transitions from state *i* to state *j*.

    Notes:
        Reviewer R3 asked for explicit smoothing handling.  The manuscript
        keeps raw counts (no pseudocounts) and shows zero-transition cells
        as blanks on the logarithmic colour scale in Figure 4A.
    """
    if not treat_as_single_chain:
        raise NotImplementedError("Segmented-chain handling not implemented.")

    if n_states is None:
        n_states = int(IDX.max()) + 1

    transitions = np.zeros((n_states, n_states), dtype=float)
    for t in range(len(IDX) - 1):
        transitions[IDX[t], IDX[t + 1]] += 1.0

    if smoothing > 0.0:
        transitions = transitions + smoothing

    return transitions


def marginal_rates(IDX: np.ndarray, n_states: int | None = None) -> np.ndarray:
    """Per-state marginal probability (base rate) over the full sequence.

    Args:
        IDX:      Integer label array.
        n_states: Number of distinct labels; inferred when None.

    Returns:
        1-D array of length ``n_states`` summing to 1.0.
    """
    if n_states is None:
        n_states = int(IDX.max()) + 1
    counts = np.bincount(IDX, minlength=n_states).astype(float)
    return counts / counts.sum()


def conditional_transition_matrix(transitions: np.ndarray) -> np.ndarray:
    """Row-normalise a raw transition-count matrix.

    Args:
        transitions: ``(n_states, n_states)`` raw counts.

    Returns:
        Same shape; each row sums to 1.0.  Rows whose source state never
        appears are left as zeros (rather than NaN-filled) so downstream
        log-scale plotting does not break.
    """
    row_sums = transitions.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(row_sums > 0, transitions / row_sums, 0.0)
    return probs


def flag_above_below_baseline(
    conditional: np.ndarray,
    marginals: np.ndarray,
    factor: float = 1.0,
) -> np.ndarray:
    """Compare each transition probability to its target state's base rate.

    For every cell ``(i, j)``, the value ``P(j | i)`` is compared to
    ``factor * P(j)``.  This is the operation that draws the up- and
    down-pointing triangles in Figure 4A of the manuscript.

    Args:
        conditional: Row-normalised transition matrix from
                     :func:`conditional_transition_matrix`.
        marginals:   Per-state base rates from :func:`marginal_rates`.
        factor:      Multiplier applied to ``P(j)`` before comparison.
                     Default 1.0.

    Returns:
        Integer array same shape as ``conditional``:
          ``+1`` where ``P(j|i) > factor * P(j)`` (above baseline),
          ``-1`` where ``P(j|i) < factor * P(j)`` (below baseline),
          ``0`` otherwise (equal, or both zero).
    """
    threshold = factor * marginals[np.newaxis, :]
    flags = np.zeros_like(conditional, dtype=int)
    flags[conditional > threshold] = 1
    flags[(conditional < threshold) & (conditional > 0)] = -1
    return flags


def expected_bout_duration(
    conditional: np.ndarray,
    fps: float = FPS,
) -> np.ndarray:
    """Geometric expectation of bout duration from self-transition probabilities.

    Under a first-order Markov assumption with self-transition
    probability ``p = P(i | i)``, bout length is geometrically distributed
    with mean ``1 / (1 - p)`` time steps.  This helper converts those
    expected durations to seconds using ``fps``.

    Args:
        conditional: Row-normalised transition matrix.
        fps:         Sampling rate in Hz (default :data:`stag.constants.FPS`).

    Returns:
        1-D array of length ``n_states`` giving expected bout duration
        in seconds.  States with ``p == 1.0`` return ``np.inf``.
    """
    p_self = np.diag(conditional)
    with np.errstate(divide="ignore", invalid="ignore"):
        steps = np.where(p_self < 1.0, 1.0 / (1.0 - p_self), np.inf)
    return steps / float(fps)
