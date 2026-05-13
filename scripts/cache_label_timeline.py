#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.cache_label_timeline                             ║
# ║  « align (deer_id, timestamp) with the saved labels.npy »        ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Sprint 3 (sequence statistics) needs per-sample timestamps and  ║
# ║  per-sample deer_id alongside the cluster labels — for circadian ║
# ║  analyses, day/night tests, per-animal time budgets, and bout    ║
# ║  statistics within an animal's recording window.                 ║
# ║                                                                  ║
# ║  Neither lives in ``clust_data_raw_20240412.npy`` (six accel +   ║
# ║  two GPS columns only) or in the saved ``labels.npy``.  Both     ║
# ║  must be reconstructed by joining ``cluster_labels`` (FK         ║
# ║  ``acc_id``) with ``accelerometer_data`` in the SQLite DB.       ║
# ║                                                                  ║
# ║  Alignment proof — verified against the production DB:           ║
# ║    1. Each deer's ``accelerometer_data.data_id`` values form a   ║
# ║       contiguous block (deer 1: 1..8.67 M, deer 2: 8.67 M..      ║
# ║       17.08 M, …, deer 26: 213.5 M..221.97 M).                   ║
# ║    2. ``make_cluster_data.py`` builds the .npy by iterating      ║
# ║       deer_ids 1..26 in order and vstacking per-deer slices      ║
# ║       (each slice in ``data_id`` ASC order after NaN-drop).      ║
# ║    3. ``insert_cluster_labels_from_npy`` inserts labels in       ║
# ║       global ``data_id`` ASC.                                    ║
# ║                                                                  ║
# ║  Both orderings traverse the deer 1 → 26 blocks in the same      ║
# ║  sequence; within a block both walk ``data_id`` ASC.  Hence      ║
# ║  ``labels.npy[i]`` corresponds to the i-th row of                ║
# ║  ``cluster_labels ORDER BY acc_id ASC`` — and to the             ║
# ║  ``accelerometer_data`` row reached by that FK.                  ║
# ║                                                                  ║
# ║  Output: two sibling ``.npy`` arrays of length 204_554_618,      ║
# ║  alongside the ``cluster_results`` tree.  Sprint 3 modules       ║
# ║  load all three (labels, deer_ids, timestamps) and slice by      ║
# ║  per-animal indices.                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Cache per-sample (deer_id, timestamp) aligned with the saved labels.npy."""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from stag.constants import (
    DEER_DB,
    LABEL_TIMELINE_DEER_IDS,
    LABEL_TIMELINE_TIMESTAMPS,
)

# Total number of labelled samples (== rows in cluster_labels == length of
# every saved ``labels.npy``).  Hardcoded to enable single-pass memmap
# allocation; verified by the script before writing anything.
N_LABELED_SAMPLES: int = 204_554_618


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEER_DB,
                        help="SQLite DB path (default: stag.constants.DEER_DB).")
    parser.add_argument("--out-deer-ids", type=Path, default=LABEL_TIMELINE_DEER_IDS,
                        help="Output .npy for per-sample deer_id.")
    parser.add_argument("--out-timestamps", type=Path, default=LABEL_TIMELINE_TIMESTAMPS,
                        help="Output .npy for per-sample timestamps (int64 ns).")
    parser.add_argument("--chunk-rows", type=int, default=2_000_000,
                        help="Pandas read_sql chunk size (default 2 M).")
    return parser.parse_args()


def _verify_total_rows(con: sqlite3.Connection) -> None:
    n = con.execute("SELECT COUNT(*) FROM cluster_labels").fetchone()[0]
    if n != N_LABELED_SAMPLES:
        raise SystemExit(
            f"cluster_labels has {n:,} rows but this script expects "
            f"{N_LABELED_SAMPLES:,}.  Either the DB was rebuilt with a "
            f"different cohort or N_LABELED_SAMPLES needs updating."
        )


def _deer_id_ranges(con: sqlite3.Connection) -> list[tuple[int, int, int, int]]:
    """Return ``[(deer_id, data_id_lo, data_id_hi, n_labels), ...]``.

    ``n_labels`` is the count of ``cluster_labels`` rows whose
    ``acc_id`` falls within this deer's ``data_id`` block — what the
    saved ``labels.npy`` actually contains for this animal.
    """
    rows = con.execute("""
        SELECT a.deer_id,
               MIN(a.data_id), MAX(a.data_id),
               COUNT(c.id)
          FROM accelerometer_data a
     LEFT JOIN cluster_labels    c ON c.acc_id = a.data_id
      GROUP BY a.deer_id
      ORDER BY a.deer_id
    """).fetchall()
    return [(int(d), int(lo), int(hi), int(n)) for d, lo, hi, n in rows]


def _stream_one_deer(
    con: sqlite3.Connection,
    deer_id: int,
    lo: int,
    hi: int,
    chunk_rows: int,
    out_timestamps: np.ndarray,
    out_deer_ids:   np.ndarray,
    offset: int,
) -> int:
    """Write one deer's slice of the join into the memmaps; return new offset.

    Uses a ``BETWEEN`` filter on the PK ``data_id`` (fastest path —
    PK ROWID seek + index seek on cluster_labels.acc_id).  Reads in
    pandas chunks so peak memory stays well under 1 GB even for the
    largest animals.
    """
    sql = f"""
        SELECT a.NZ_DateTime
          FROM accelerometer_data a
          JOIN cluster_labels c ON c.acc_id = a.data_id
         WHERE a.data_id BETWEEN ? AND ?
      ORDER BY a.data_id
    """
    for chunk in pd.read_sql(sql, con, params=(lo, hi), chunksize=chunk_rows,
                             parse_dates=["NZ_DateTime"]):
        n = len(chunk)
        out_timestamps[offset:offset + n] = chunk["NZ_DateTime"].astype("int64").values
        out_deer_ids[offset:offset + n]   = deer_id
        offset += n
    return offset


def main() -> None:
    args = parse_args()
    args.out_deer_ids.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    _verify_total_rows(con)
    ranges = _deer_id_ranges(con)
    total = sum(n for *_, n in ranges)
    if total != N_LABELED_SAMPLES:
        raise SystemExit(
            f"Per-deer label counts sum to {total:,}, expected "
            f"{N_LABELED_SAMPLES:,}.  Refusing to write incomplete output."
        )
    print(f"DB        : {args.db}")
    print(f"cohort    : {len(ranges)} animals  ({sum(1 for _,_,_,n in ranges if n>0)} with labels)")
    print(f"total rows: {total:,}")
    print()

    # Allocate the two memmaps up front so we fail fast on disk-full.
    ts  = np.lib.format.open_memmap(
        args.out_timestamps, mode="w+", dtype=np.int64,
        shape=(N_LABELED_SAMPLES,),
    )
    did = np.lib.format.open_memmap(
        args.out_deer_ids, mode="w+", dtype=np.int8,
        shape=(N_LABELED_SAMPLES,),
    )

    offset = 0
    t0 = time.time()
    for deer_id, lo, hi, n in tqdm(ranges, desc="caching by deer"):
        if n == 0:
            continue
        offset = _stream_one_deer(
            con, deer_id, lo, hi, args.chunk_rows,
            ts, did, offset,
        )

    ts.flush()
    did.flush()
    con.close()

    if offset != N_LABELED_SAMPLES:
        raise SystemExit(
            f"Wrote {offset:,} rows, expected {N_LABELED_SAMPLES:,}.  "
            f"Output files are present but truncated — re-run.",
        )

    elapsed = time.time() - t0
    print()
    print(f"  wrote {args.out_timestamps}  ({Path(args.out_timestamps).stat().st_size / 1e9:.2f} GB)")
    print(f"  wrote {args.out_deer_ids}    ({Path(args.out_deer_ids).stat().st_size / 1e6:.1f} MB)")
    print(f"  elapsed: {elapsed/60:.1f} min")

    # ── quick sanity check ──────────────────────────────────────────
    ts2  = np.load(args.out_timestamps, mmap_mode="r")
    did2 = np.load(args.out_deer_ids,   mmap_mode="r")
    print()
    print("── sanity ──")
    print(f"  first sample : deer={did2[0]:2d}   t={pd.Timestamp(ts2[0])}")
    print(f"  last  sample : deer={did2[-1]:2d}  t={pd.Timestamp(ts2[-1])}")
    n_per_deer = np.bincount(did2)
    for d, count in enumerate(n_per_deer):
        if count == 0:
            continue
        print(f"  deer {d:2d} : {count:,d} samples")


if __name__ == "__main__":
    main()
