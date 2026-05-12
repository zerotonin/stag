#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.preprocess_clustering_data                       ║
# ║  « reproduce the 2024 SLURM pipeline's normalisation »           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The production SLURM clustering in 2024 (Aoraki                ║
# ║  clustering_script.py) performed three preprocessing steps      ║
# ║  inside the per-job script itself:                              ║
# ║                                                                  ║
# ║    1. Slice the 8-column raw feature matrix to the first 6      ║
# ║       columns (the accelerometer axes; GPS speed and tortuosity ║
# ║       were dropped per manuscript §2.2).                         ║
# ║    2. Clip column 5 (Ear_Z) to ±7.99 — works around an           ║
# ║       ingestion bug that produced nine astronomically large     ║
# ║       outliers in that column only.                              ║
# ║    3. Apply cuML's MaxAbsScaler — divide each column by its     ║
# ║       absolute maximum, mapping the data to [-1, 1] per         ║
# ║       column.  Crucially, NOT StandardScaler: the historical    ║
# ║       script's StandardScaler line is commented out, and using  ║
# ║       it produces centroids in a completely different scale     ║
# ║       (column-5's small σ ≈ 0.39 would stretch the data to     ║
# ║       ±20+, breaking the partition geometry).                   ║
# ║                                                                  ║
# ║  This script bakes all three steps into a single .npy that     ║
# ║  downstream tooling can consume directly: feed it to            ║
# ║  `stag.clustering.kmeans --no-rescale` and the cuML KMeans      ║
# ║  fit reproduces the 2024 centroids bit-for-bit (modulo GPU      ║
# ║  non-determinism in the few percent of high-k positions that   ║
# ║  sit near degenerate local optima).                             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Slice + col-5-clip + per-column MaxAbsScaler, matching 2024 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from stag.constants import MAXABS_CLUSTERING_INPUT, RAW_CLUSTERING_INPUT

# Six-column slice the 2024 production pipeline used.  See the
# manuscript §2.2: "the six accelerometer axes formed the final
# feature set".  Columns 6-7 of the raw file are GPS-derived
# (abs_speed_mPs, tortuosity) and were excluded from clustering.
ACCEL_COLS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

# Column 5 in the raw file (Ear_Z) carries nine astronomically large
# outliers (up to 2 × 10²⁴⁵) that bypassed the upstream sensor-cap
# clip.  The historical clustering_script.py clipped them to ±7.99
# with a BADHACK comment; we reproduce the same fix here.
COL5_CLIP_RANGE: float = 7.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_CLUSTERING_INPUT,
                        help="Raw 8-column .npy (default: "
                             "stag.constants.RAW_CLUSTERING_INPUT).")
    parser.add_argument("--out", type=Path, default=MAXABS_CLUSTERING_INPUT,
                        help="Output 6-column MaxAbs-scaled .npy path "
                             "(default: stag.constants.MAXABS_CLUSTERING_INPUT).")
    parser.add_argument("--chunk-rows", type=int, default=2_000_000,
                        help="Rows per streaming chunk (default 2e6 ≈ 96 MB / 6 cols).")
    parser.add_argument("--cols", type=int, nargs="+", default=list(ACCEL_COLS),
                        help="Column indices to keep (default: the six accel axes).")
    parser.add_argument("--col5-clip", type=float, default=COL5_CLIP_RANGE,
                        help="Symmetric clip applied to column 5 (Ear_Z) before "
                             "computing the per-column max abs.  Default ±7.99 "
                             "matching the historical clustering_script.py.")
    return parser.parse_args()


def _clip_col5_inplace(chunk: np.ndarray, cols: list[int], clip_range: float) -> int:
    """Clip column 5 (if present in ``cols``) of ``chunk`` to ±clip_range.

    Returns the number of values that were clamped.  Other columns
    are untouched — this matches the historical pipeline which only
    applied the BADHACK to Ear_Z, not the entire matrix.
    """
    if 5 not in cols:
        return 0
    col5_idx = cols.index(5)
    out_of_range = np.abs(chunk[:, col5_idx]) > clip_range
    if out_of_range.any():
        chunk[:, col5_idx] = np.clip(chunk[:, col5_idx], -clip_range, clip_range)
    # Also catch non-finite values that survived the magnitude check.
    bad = ~np.isfinite(chunk[:, col5_idx])
    if bad.any():
        chunk[bad, col5_idx] = 0.0
    return int(out_of_range.sum()) + int(bad.sum())


def _streaming_maxabs(
    raw: np.ndarray, cols: list[int],
    chunk_rows: int, col5_clip: float,
) -> tuple[np.ndarray, int]:
    """One pass over the mmap'd raw matrix → per-column max abs.

    Column 5 is clipped to ±``col5_clip`` before its max abs is
    measured; other columns are left as-is.  Returns the per-column
    max-abs vector and the total count of col-5 clip events.
    """
    n_cols = len(cols)
    maxabs = np.zeros(n_cols, dtype=np.float64)
    col5_clip_events = 0

    n_rows = raw.shape[0]
    for start in tqdm(range(0, n_rows, chunk_rows), desc="pass 1/2 (max abs)"):
        end = min(start + chunk_rows, n_rows)
        chunk = np.asarray(raw[start:end, cols], dtype=np.float64).copy()
        col5_clip_events += _clip_col5_inplace(chunk, cols, col5_clip)
        maxabs = np.maximum(maxabs, np.abs(chunk).max(axis=0))

    return maxabs, col5_clip_events


def _streaming_scale_write(
    raw: np.ndarray, cols: list[int], maxabs: np.ndarray,
    out_path: Path, chunk_rows: int, col5_clip: float,
) -> None:
    """Second pass: clip col 5, divide by per-column max abs, write."""
    n_rows = raw.shape[0]
    n_cols = len(cols)
    safe = np.where(maxabs > 0, maxabs, 1.0)

    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float64,
        shape=(n_rows, n_cols),
    )
    for start in tqdm(range(0, n_rows, chunk_rows), desc="pass 2/2 (write)"):
        end = min(start + chunk_rows, n_rows)
        chunk = np.asarray(raw[start:end, cols], dtype=np.float64).copy()
        _clip_col5_inplace(chunk, cols, col5_clip)
        out[start:end] = chunk / safe
    out.flush()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.raw, mmap_mode="r")
    print(f"Input         : {args.raw}  shape={raw.shape}  dtype={raw.dtype}")
    print(f"Keep cols     : {args.cols}")
    print(f"Col-5 clip    : ±{args.col5_clip}")
    print(f"Output        : {args.out}  shape=({raw.shape[0]}, {len(args.cols)})  dtype=float64")
    print()

    maxabs, col5_clip_events = _streaming_maxabs(
        raw, args.cols, args.chunk_rows, args.col5_clip,
    )
    print()
    print("Per-column max abs (used as MaxAbsScaler divisors):")
    for i, (col, m) in enumerate(zip(args.cols, maxabs)):
        print(f"  col {col} : {m:.6f}")
    print(f"Col-5 clip events: {col5_clip_events}  "
          f"(historical script reports 9 for clust_data_raw_20240412.npy)")
    print()

    _streaming_scale_write(
        raw, args.cols, maxabs, args.out, args.chunk_rows, args.col5_clip,
    )

    # Provenance sidecar so future-anyone can recover the absolute
    # values from the scaled .npy if needed.
    csv_path = args.out.with_suffix(".maxabs.csv")
    header = ",".join(f"col{c}" for c in args.cols)
    np.savetxt(csv_path, maxabs[None, :], delimiter=",",
               header=header, comments="")
    print(f"Wrote: {args.out}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
