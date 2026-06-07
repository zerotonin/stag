#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.merge_silhouette_extension                       ║
# ║  « merge SLURM-array silhouette CSVs and re-render Figure 2D »   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  After ``slurm/silhouette_extension_array.sh`` finishes, each    ║
# ║  task has written one row to                                     ║
# ║  ``results/sprint1/tables/silhouette_ext_delSize<X>_k<Y>.csv``.  ║
# ║  This script:                                                    ║
# ║                                                                  ║
# ║    1. Concatenates every per-task CSV into a single tall table.  ║
# ║    2. Merges it with the existing k <= 20 silhouette table.      ║
# ║    3. Re-renders the silhouette + Kneedle elbow figure with the  ║
# ║       full k = 2..50 range.                                      ║
# ║                                                                  ║
# ║  Run with no arguments from the repo root after the array is     ║
# ║  complete; defaults match the SLURM script's output paths.       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Merge SLURM-array per-task silhouette CSVs and re-render the figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from kneed import KneeLocator

from stag.constants import RESULTS_DIR_DEFAULT, apply_figure_defaults, save_figure

PALETTE: dict[int, str] = {
    0:  "#EECCC9",
    10: "#DB97AA",
    25: "#A75491",
    50: "#291E38",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir", type=Path,
        default=RESULTS_DIR_DEFAULT / "sprint1" / "tables",
        help="Directory holding per-task silhouette_ext_*.csv files.",
    )
    parser.add_argument(
        "--existing-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "tables" / "silhouette_per_delSize_k.csv",
        help="Existing k <= 20 silhouette CSV from the local run.",
    )
    parser.add_argument(
        "--inertia-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "figures"
                / "figure2_internal_metrics_4reductions.csv",
        help="Combined per-(delSize,k) inertia summary for the elbow panel.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR_DEFAULT,
        help="Root for outputs (figures land in <output-dir>/figures).",
    )
    parser.add_argument("--chosen-k", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1) Stitch the per-task CSVs.
    ext_files = sorted(args.tables_dir.glob("silhouette_ext_*.csv"))
    if not ext_files:
        raise SystemExit(
            f"No silhouette_ext_*.csv files under {args.tables_dir}; "
            "did the SLURM array complete?",
        )
    print(f"Merging {len(ext_files)} per-task CSVs:")
    for f in ext_files:
        print(f"  {f.name}")
    ext_df = pd.concat([pd.read_csv(f) for f in ext_files], ignore_index=True)

    # 2) Combine with the existing k <= 20 table.
    if args.existing_csv.exists():
        existing_df = pd.read_csv(args.existing_csv)
        print(f"\nLoaded existing low-k table: {len(existing_df)} rows")
        full_df = pd.concat([existing_df, ext_df], ignore_index=True)
        full_df = full_df.drop_duplicates(
            subset=["delSize", "k"], keep="last",
        ).sort_values(["delSize", "k"]).reset_index(drop=True)
    else:
        full_df = ext_df

    out_csv = tables_dir / "silhouette_per_delSize_k_extended.csv"
    full_df.to_csv(out_csv, index=False)
    print(f"\nWrote merged silhouette table: {out_csv}")

    # 3) Recompute Kneedle elbows on the inertia data (unchanged scope).
    inertia_summary = pd.read_csv(args.inertia_csv)
    elbows: dict[int, int] = {}
    for ds, sub in inertia_summary.groupby("delSize"):
        sub = sub.sort_values("k").dropna(subset=["inertia"])
        if len(sub) >= 3:
            knee = KneeLocator(
                sub["k"], sub["inertia"],
                curve="convex", direction="decreasing", S=1.0,
            )
            if knee.knee is not None:
                elbows[int(ds)] = int(knee.knee)
    print(f"\nKneedle elbows per delSize: {elbows}")

    # 4) Render — 2x1 vertical layout, shared x-axis so the silhouette
    # curve and the inertia curve are read against the same k grid.
    apply_figure_defaults()
    fig, (ax_sil, ax_W) = plt.subplots(
        2, 1, figsize=(7.5, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    for ds, colour in PALETTE.items():
        sub = full_df[full_df["delSize"] == ds].sort_values("k")
        if sub.empty:
            continue
        ax_sil.fill_between(
            sub["k"], sub["silhouette_low"], sub["silhouette_high"],
            facecolor=colour, alpha=0.30, linewidth=0,
        )
        ax_sil.plot(
            sub["k"], sub["silhouette"], "-o", color=colour, markersize=4,
            linewidth=1.4, label=f"delSize {ds} %",
        )
    ax_sil.set_title("(C) Silhouette per leave-out reduction (k = 2 .. 50)")
    ax_sil.set_ylabel(r"Mean silhouette ($\bar{s}$)")
    ax_sil.axvline(args.chosen_k, color="black", linestyle="--",
                   linewidth=0.8, alpha=0.5)
    ax_sil.legend(fontsize="small", loc="best", frameon=False)

    for ds, colour in PALETTE.items():
        sub = inertia_summary[inertia_summary["delSize"] == ds].sort_values("k")
        if sub.empty:
            continue
        ax_W.fill_between(
            sub["k"], sub["inertia_low"], sub["inertia_high"],
            facecolor=colour, alpha=0.30, linewidth=0,
        )
        ax_W.plot(
            sub["k"], sub["inertia"], "-o", color=colour, markersize=4,
            linewidth=1.4, label=f"delSize {ds} %",
        )
        if ds in elbows:
            k_e = elbows[ds]
            row_e = sub[sub["k"] == k_e]
            if not row_e.empty:
                y_e = float(row_e["inertia"].iloc[0])
                ax_W.scatter(
                    [k_e], [y_e],
                    s=100, facecolor="white", edgecolor=colour, linewidth=2.0,
                    zorder=6, label=f"Kneedle k = {k_e} (delSize {ds} %)",
                )
    ax_W.set_title("(D) Inertia / Kneedle elbow per leave-out reduction")
    ax_W.set_xlabel("k")
    ax_W.set_ylabel(r"$W(k)$")
    ax_W.axvline(args.chosen_k, color="black", linestyle="--",
                 linewidth=0.8, alpha=0.5)
    ax_W.legend(fontsize="x-small", loc="best", frameon=False)

    # Shared x-axis: set ticks once on the bottom panel (sharex
    # propagates).
    ax_W.set_xticks([2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50])
    ax_W.tick_params(axis="x", labelsize="x-small")

    fig.tight_layout()
    save_figure(
        fig, "silhouette_elbow_per_delSize_extended", figures_dir,
        data=full_df,
    )
    print(f"\nWrote {figures_dir}/silhouette_elbow_per_delSize_extended.{{svg,png,csv}}")


if __name__ == "__main__":
    main()
