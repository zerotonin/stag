#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.sync_db_labels                                   ║
# ║  « overwrite cluster_labels.label with the current k=8 labels »  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The historical ``cluster_labels`` table on HCS was populated   ║
# ║  from a *pre-MaxAbs* clustering run with different cluster IDs  ║
# ║  than the current k = 8 saved labels.  Any DB-side query that   ║
# ║  joins on ``cluster_labels.label`` therefore returns stale      ║
# ║  cluster assignments.                                            ║
# ║                                                                  ║
# ║  This script does three things, in order:                       ║
# ║                                                                  ║
# ║    1. Archives the existing ``cluster_labels`` rows into a       ║
# ║       new table ``cluster_labels_pre_maxabs_run`` (full copy,    ║
# ║       same schema, indexed on ``acc_id``).  Nothing is lost.    ║
# ║    2. Overwrites ``cluster_labels.label`` row-by-row from the    ║
# ║       current k = 8 ``labels.npy``.  Row alignment uses          ║
# ║       ``cluster_labels.id`` (PK, autoincrement) which matches    ║
# ║       the ``labels.npy`` row index because the original loader   ║
# ║       (``insert_cluster_labels_from_npy``) inserted in           ║
# ║       ``data_id`` ASC global order — the same order              ║
# ║       ``make_cluster_data.py`` used to build the .npy.           ║
# ║    3. Records the swap in a small ``table_provenance`` table     ║
# ║       so future-anyone reads the DB and sees what happened.     ║
# ║                                                                  ║
# ║  Default behaviour is a *dry-run* — pass ``--execute`` to        ║
# ║  actually mutate the DB.  Run on the local copy only; never on  ║
# ║  the HCS originals.                                              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Overwrite cluster_labels.label with current k=8 .npy; archive the old run."""

from __future__ import annotations

import argparse
import datetime
import sqlite3
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from stag.constants import CLUSTER_RESULTS_DIR, DEER_DB

LEGACY_TABLE_DEFAULT: str = "cluster_labels_pre_maxabs_run"
PROVENANCE_TABLE: str = "table_provenance"
DEFAULT_K: int = 8
DEFAULT_BATCH: int = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEER_DB,
                        help="SQLite DB (default: stag.constants.DEER_DB).")
    parser.add_argument("--labels", type=Path, default=None,
                        help="Path to the labels .npy to write.  Defaults to "
                             "the first match in "
                             "cluster_results/deer6raw/delSize_0/k_8/labels/*.npy.")
    parser.add_argument("--legacy-table", type=str, default=LEGACY_TABLE_DEFAULT,
                        help=f"Name of the archive table (default {LEGACY_TABLE_DEFAULT}).")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"Rows per UPDATE transaction (default {DEFAULT_BATCH}).")
    parser.add_argument("--execute", action="store_true",
                        help="Actually mutate the DB (default is dry-run).")
    return parser.parse_args()


def _resolve_labels_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = sorted(
        (CLUSTER_RESULTS_DIR / "delSize_0" / f"k_{DEFAULT_K}" / "labels").glob("*.npy")
    )
    if not candidates:
        raise SystemExit(
            f"No k={DEFAULT_K} labels found under "
            f"{CLUSTER_RESULTS_DIR / 'delSize_0' / f'k_{DEFAULT_K}' / 'labels'}.  "
            f"Pass --labels explicitly."
        )
    # All 50 delPosP files at delSize_0 are bit-identical (random_state=0,
    # shrink_data fast-path).  Any of them works; use the first.
    return candidates[0]


def _ensure_provenance_table(con: sqlite3.Connection) -> None:
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {PROVENANCE_TABLE} (
            table_name  TEXT PRIMARY KEY,
            description TEXT,
            created_at  TEXT,
            source      TEXT,
            n_rows      INTEGER
        )
    """)


def _record_provenance(
    con: sqlite3.Connection, table: str, description: str, source: str, n_rows: int,
) -> None:
    con.execute(f"""
        INSERT OR REPLACE INTO {PROVENANCE_TABLE}
            (table_name, description, created_at, source, n_rows)
        VALUES (?, ?, ?, ?, ?)
    """, (table, description, datetime.datetime.utcnow().isoformat() + "Z", source, n_rows))


def _archive_existing(con: sqlite3.Connection, archive_name: str) -> int:
    """Copy ``cluster_labels`` into ``archive_name`` with its index.  Returns row count."""
    exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (archive_name,),
    ).fetchone()
    if exists:
        raise SystemExit(
            f"Archive table {archive_name!r} already exists.  Refusing to overwrite — "
            f"either pass a different --legacy-table or DROP the existing one manually."
        )

    print(f"  archiving cluster_labels → {archive_name} ...")
    con.execute(f"""
        CREATE TABLE {archive_name} AS
            SELECT * FROM cluster_labels
    """)
    con.execute(f"CREATE INDEX ix_{archive_name}_acc_id ON {archive_name}(acc_id)")
    n = con.execute(f"SELECT COUNT(*) FROM {archive_name}").fetchone()[0]
    print(f"  archived {n:,} rows into {archive_name}")
    return int(n)


def _bulk_update_labels(
    con: sqlite3.Connection,
    labels: np.ndarray,
    batch: int,
) -> None:
    """UPDATE cluster_labels.label = labels[id-1] in batched transactions.

    Uses the PK ``id`` for row addressing because it matches the
    insertion order of the original loader (``insert_cluster_labels_from_npy``
    walks accelerometer_data in ``data_id`` ASC and inserts one
    cluster_labels row per accel row that has a label).  ``id = i+1``
    therefore corresponds to ``labels[i]``.
    """
    n = labels.size
    cur = con.cursor()

    t0 = time.time()
    with tqdm(total=n, desc="UPDATE cluster_labels.label", unit="row",
              unit_scale=True) as pbar:
        for start in range(0, n, batch):
            end = min(start + batch, n)
            cur.execute("BEGIN")
            cur.executemany(
                "UPDATE cluster_labels SET label = ? WHERE id = ?",
                ((int(labels[i]), i + 1) for i in range(start, end)),
            )
            con.commit()
            pbar.update(end - start)
    print(f"  elapsed: {(time.time() - t0)/60:.1f} min")


def _verify(con: sqlite3.Connection, labels: np.ndarray) -> None:
    print()
    print("── verification ──")
    n = con.execute("SELECT COUNT(*) FROM cluster_labels").fetchone()[0]
    print(f"  cluster_labels rows         : {n:,}  (expected {labels.size:,})")
    if n != labels.size:
        raise SystemExit("Row count mismatch after update.")

    # 10 random spot checks
    rng = np.random.default_rng(0)
    sample_ids = rng.integers(1, labels.size + 1, size=10)
    sample_ids.sort()
    print("  spot checks (random ids):")
    for row_id in sample_ids:
        db_label = con.execute(
            "SELECT label FROM cluster_labels WHERE id = ?", (int(row_id),),
        ).fetchone()[0]
        npy_label = int(labels[row_id - 1])
        match = "✓" if db_label == npy_label else "✗ MISMATCH"
        print(f"    id={row_id:>10}  db={db_label}  npy={npy_label}  {match}")

    # Histogram comparison.
    print()
    print("  PM histograms (DB vs npy):")
    db_hist = dict(con.execute(
        "SELECT label, COUNT(*) FROM cluster_labels GROUP BY label ORDER BY label"
    ).fetchall())
    npy_hist = np.bincount(labels)
    print(f"    {'PM':>3}  {'DB':>14}  {'npy':>14}  {'match':>6}")
    for pm in range(int(npy_hist.size)):
        db_n = int(db_hist.get(pm, 0))
        npy_n = int(npy_hist[pm])
        match = "✓" if db_n == npy_n else "✗"
        print(f"    {pm:>3}  {db_n:14,d}  {npy_n:14,d}  {match:>6}")


def main() -> None:
    args = parse_args()

    labels_path = _resolve_labels_path(args.labels)
    labels = np.load(labels_path)
    print(f"DB           : {args.db}")
    print(f"labels file  : {labels_path}")
    print(f"  shape: {labels.shape}  dtype: {labels.dtype}  "
          f"unique: {sorted(np.unique(labels).tolist())}")
    print(f"archive table: {args.legacy_table}")
    print(f"mode         : {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print()

    if not args.execute:
        print("Dry-run — no changes made.  Re-run with --execute to perform the swap.")
        return

    # ``isolation_level=None`` ⇒ autocommit; we drive transactions
    # explicitly with BEGIN/COMMIT in the bulk-update loop.  Required
    # because PRAGMA journal_mode and PRAGMA synchronous cannot be
    # changed inside a transaction, and python's sqlite3 default
    # opens an implicit transaction on the first DML/DDL.
    con = sqlite3.connect(str(args.db), isolation_level=None)
    try:
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA synchronous = NORMAL")
        _ensure_provenance_table(con)
        n_arch = _archive_existing(con, args.legacy_table)
        _record_provenance(
            con, args.legacy_table,
            description=(
                "Pre-MaxAbs cluster_labels.  Pre-2026-05-13 clustering run "
                "(z-score era, different cluster IDs than the canonical "
                "k=8 saved labels).  Kept for provenance; do NOT join on .label."
            ),
            source="archived from cluster_labels in sync_db_labels.py",
            n_rows=n_arch,
        )

        _bulk_update_labels(con, labels, args.batch)
        _record_provenance(
            con, "cluster_labels",
            description=(
                f"Current k={DEFAULT_K} labels from the MaxAbs production "
                f"refit (2026-05-13).  Source .npy aligned to cluster_labels.id "
                f"in insertion order (acc_id ASC)."
            ),
            source=str(labels_path),
            n_rows=int(labels.size),
        )

        con.commit()
        _verify(con, labels)

        print()
        print("  ANALYZE cluster_labels ...")
        con.execute("ANALYZE cluster_labels")
        con.commit()
        print("  done.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
