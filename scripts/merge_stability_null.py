#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.merge_stability_null                             ║
# ║  « aggregate the R3 stability-null block CSVs into a band »      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  After the SLURM array completes, every                          ║
# ║  ``stability_null_seed<SS>_k<KK>.csv`` holds one row per          ║
# ║  k-means fit (typically 20).  This script:                       ║
# ║                                                                  ║
# ║    1. Concatenates every per-task CSV.                           ║
# ║    2. Within each (surrogate_seed, k) block, takes the median    ║
# ║       Hungarian-matched centroid drift across the 20 fits        ║
# ║       (one value per surrogate draw).                            ║
# ║    3. Across the 10 surrogate seeds, computes the median + IQR   ║
# ║       at each k -> ``instability_null``, ``instability_null_     ║
# ║       low``, ``instability_null_high``.                          ║
# ║                                                                  ║
# ║  The output CSV is consumed by                                   ║
# ║  ``scripts/replot_internal_metrics_4reductions.py`` to overlay   ║
# ║  the d-ceiling band on Panel B.                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Aggregate stability-null per-task CSVs into the d-ceiling band table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stag.constants import RESULTS_DIR_DEFAULT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir", type=Path,
        default=RESULTS_DIR_DEFAULT / "sprint1" / "tables"
                / "stability_null_uniform",
        help="Directory holding the per-(seed, k) stability_null CSVs.",
    )
    parser.add_argument(
        "--output-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "figures"
                / "figure2_stability_null_uniform.csv",
        help="Destination band CSV (k, instability_null + IQR bounds).",
    )
    parser.add_argument(
        "--per-seed-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "sprint1" / "tables"
                / "stability_null_uniform_per_seed.csv",
        help="Intermediate per-(seed, k) median table (kept for audit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csvs = sorted(args.tables_dir.glob("stability_null_seed*_k*.csv"))
    if not csvs:
        raise SystemExit(
            f"No stability_null_seed*_k*.csv files under {args.tables_dir}; "
            "did the SLURM array complete?",
        )

    print(f"Merging {len(csvs)} per-task CSVs from {args.tables_dir}")
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    print(
        f"  rows = {len(df)}, "
        f"surrogate_seeds = {sorted(df['surrogate_seed'].unique())}, "
        f"k range = {df['k'].min()}..{df['k'].max()}",
    )

    finite = df.dropna(subset=["instability_null"])

    # Per-(seed, k): median all metrics across the 20 k-means fits per
    # surrogate draw.  CH / silhouette / inertia rows are NaN-free in
    # the per-task CSV (computed on every fit including the reference);
    # instability carries NaN for the reference fit so the median
    # naturally skips it.
    available_per_fit: list[tuple[str, str]] = [
        ("instability_null", "instability_null"),
        ("ch",               "ch_null"),
    ]
    if "silhouette" in finite.columns:
        available_per_fit.append(("silhouette", "silhouette_null"))
    if "inertia" in finite.columns:
        available_per_fit.append(("inertia", "inertia_null"))

    agg_kwargs = {out_col: (in_col, "median") for in_col, out_col in available_per_fit}
    per_seed = (
        finite.groupby(["surrogate_seed", "k"])
              .agg(**agg_kwargs)
              .reset_index()
    )
    args.per_seed_csv.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.per_seed_csv, index=False)
    print(f"Wrote per-seed table: {args.per_seed_csv}")

    # Cross-seed band: median + IQR per metric.
    def _band_for(col: str, out_prefix: str) -> pd.DataFrame:
        return (
            per_seed.groupby("k")[col]
                    .agg(**{
                        out_prefix:           "median",
                        f"{out_prefix}_low":  lambda s: float(np.quantile(s, 0.25)),
                        f"{out_prefix}_high": lambda s: float(np.quantile(s, 0.75)),
                    })
                    .reset_index()
        )

    out_cols = [out for _, out in available_per_fit]
    band = _band_for(out_cols[0], out_cols[0])
    for out in out_cols[1:]:
        band = band.merge(_band_for(out, out), on="k", how="outer")
    band = band.sort_values("k")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    band.to_csv(args.output_csv, index=False)
    print(f"Wrote band CSV: {args.output_csv}")
    print(band.to_string(index=False))


if __name__ == "__main__":
    main()
