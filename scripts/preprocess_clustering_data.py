#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.preprocess_clustering_data                       ║
# ║  « reproduce the z-scored 6-column feature matrix »              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The SLURM clustering on Aoraki used six accelerometer axes      ║
# ║  (Head_{X,Y,Z} + Ear_{X,Y,Z}) — GPS speed and tortuosity were    ║
# ║  excluded because the 0.5 Hz GPS sampling rate could not resolve ║
# ║  fine head/ear motion.  The raw .npy archive still contains      ║
# ║  eight columns, so we slice + z-score here in one streaming      ║
# ║  pass and write the result back as a fresh .npy.  Downstream     ║
# ║  silhouette / inertia code then sees a clean array that lines    ║
# ║  up exactly with the saved 6-D centroids and labels.             ║
# ║                                                                  ║
# ║  Streaming: two passes over the raw file with Welford-style      ║
# ║  online statistics, never materialising more than a chunk of     ║
# ║  rows at a time, so the workstation's 62 GB RAM is never a       ║
# ║  bottleneck.                                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Slice raw clustering input to 6 columns and z-score to disk."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from stag.constants import RAW_CLUSTERING_INPUT, ZSCORED_CLUSTERING_INPUT

# Six-column slice the production SLURM clustering used.  See the
# manuscript §2.2: "the six accelerometer axes formed the final feature
# set".  The remaining columns (abs_speed_mPs, tortuosity) are kept on
# disk in the raw file but are NOT used at fit time.
ACCEL_COLS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_CLUSTERING_INPUT,
                        help="Raw 8-column .npy (default: stag.constants.RAW_CLUSTERING_INPUT).")
    parser.add_argument("--out", type=Path, default=ZSCORED_CLUSTERING_INPUT,
                        help="Output 6-column z-scored .npy path "
                             "(default: stag.constants.ZSCORED_CLUSTERING_INPUT).")
    parser.add_argument("--chunk-rows", type=int, default=2_000_000,
                        help="Rows per streaming chunk (default 2e6 ≈ 96 MB / 6 cols).")
    parser.add_argument("--cols", type=int, nargs="+", default=list(ACCEL_COLS),
                        help="Column indices to keep (default: the six accel axes).")
    parser.add_argument("--clip-range", type=float, default=8.0,
                        help="Per-column saturation limit applied before "
                             "z-scoring (default ±8).  Matches the upstream "
                             "preprocessing protocol where raw accelerometer "
                             "values were clipped to the ±8 g sensor range; "
                             "any value beyond that is a corruption from an "
                             "earlier ingestion bug (nine such samples found "
                             "in column 5 of clust_data_raw_20240412.npy with "
                             "magnitudes up to 2 × 10²⁴⁵).")
    return parser.parse_args()


def _clip_chunk(chunk: np.ndarray, clip_range: float) -> tuple[np.ndarray, np.ndarray]:
    """Clip ``|x| > clip_range`` to ±clip_range; replace any remaining
    non-finite values with 0.  Returns the clipped chunk and the
    per-column count of values that were modified.

    Closes the gap from the upstream clipping protocol: most samples
    were already in ±8 by design, but nine col-5 samples escaped and
    carry magnitudes up to 2 × 10²⁴⁵.  We re-apply the cap here so the
    z-score statistics are not poisoned.
    """
    over_range = np.abs(chunk) > clip_range
    non_finite = ~np.isfinite(chunk)
    modified = over_range | non_finite
    out = np.clip(chunk, -clip_range, clip_range)
    # np.clip preserves NaN; replace those explicitly.
    if non_finite.any():
        out = np.where(non_finite, 0.0, out)
    return out, modified.sum(axis=0).astype(np.int64)


def _streaming_mu_sigma(
    raw: np.ndarray, cols: list[int], chunk_rows: int,
    clip_range: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One pass over the mmap'd raw matrix → per-column mean and std.

    Each chunk is clipped to ±``clip_range`` before being merged into
    the running statistics via Welford's online algorithm.  Returns
    the population mean, the sample std (ddof=0, matching
    ``sklearn.preprocessing.StandardScaler``), and the per-column
    count of values that hit the saturation limit.
    """
    n_features = len(cols)
    count = 0
    mean = np.zeros(n_features, dtype=np.float64)
    M2 = np.zeros(n_features, dtype=np.float64)
    clipped = np.zeros(n_features, dtype=np.int64)

    n_rows = raw.shape[0]
    for start in tqdm(range(0, n_rows, chunk_rows), desc="pass 1/2 (mu, sigma)"):
        end = min(start + chunk_rows, n_rows)
        chunk_raw = np.asarray(raw[start:end, cols], dtype=np.float64)
        chunk, chunk_clipped = _clip_chunk(chunk_raw, clip_range)
        clipped += chunk_clipped

        m = end - start
        chunk_mean = chunk.mean(axis=0)
        chunk_var_times_m = ((chunk - chunk_mean) ** 2).sum(axis=0)

        # Chan/Welford merge.
        delta = chunk_mean - mean
        new_count = count + m
        mean = mean + delta * (m / new_count)
        M2 = M2 + chunk_var_times_m + delta**2 * (count * m / new_count)
        count = new_count

    std = np.sqrt(M2 / count)
    return mean, std, clipped


def _streaming_zscore_write(
    raw: np.ndarray, cols: list[int], mu: np.ndarray, sigma: np.ndarray,
    out_path: Path, chunk_rows: int, clip_range: float,
) -> None:
    """Second pass: clip → z-score → write to a new .npy on disk.

    Uses the same clip-to-±``clip_range`` step as the statistics pass
    so the output is fully consistent with the (mu, sigma) computed
    in pass 1.
    """
    n_rows = raw.shape[0]
    n_features = len(cols)
    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float64,
        shape=(n_rows, n_features),
    )
    # Guard against div-by-zero columns (constant features).  Real data
    # should never trigger this but we set a placeholder sigma=1.0 so
    # a constant column z-scores to 0 rather than NaN.
    safe_sigma = np.where(sigma > 0, sigma, 1.0)
    for start in tqdm(range(0, n_rows, chunk_rows), desc="pass 2/2 (write)"):
        end = min(start + chunk_rows, n_rows)
        chunk_raw = np.asarray(raw[start:end, cols], dtype=np.float64)
        chunk, _ = _clip_chunk(chunk_raw, clip_range)
        out[start:end] = (chunk - mu) / safe_sigma
    out.flush()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.raw, mmap_mode="r")
    print(f"Input      : {args.raw}  shape={raw.shape}  dtype={raw.dtype}")
    print(f"Keep cols  : {args.cols}")
    print(f"Clip range : ±{args.clip_range}")
    print(f"Output     : {args.out}  shape=({raw.shape[0]}, {len(args.cols)})  dtype=float64")
    print()

    mu, sigma, clipped = _streaming_mu_sigma(
        raw, args.cols, args.chunk_rows, args.clip_range,
    )
    print()
    print("Per-column mu:           ", mu.round(4))
    print("Per-column sigma:        ", sigma.round(4))
    print("Per-column clip events:  ", clipped)
    total_rows = raw.shape[0]
    print(f"(out of {total_rows:,} rows per column)")
    print()

    _streaming_zscore_write(
        raw, args.cols, mu, sigma, args.out, args.chunk_rows, args.clip_range,
    )

    # Save the mu/sigma alongside the output so the rebuttal can show
    # the exact normalisation parameters used.
    csv_path = args.out.with_suffix(".musigma.csv")
    np.savetxt(
        csv_path,
        np.column_stack([mu, sigma]),
        delimiter=",", header="mu,sigma", comments="",
    )
    print(f"Wrote: {args.out}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
