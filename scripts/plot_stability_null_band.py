#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.plot_stability_null_band                         ║
# ║  « R3 d-ceiling preview from the merged null band »              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Reads the per-k stability-null band CSV emitted by              ║
# ║  scripts/merge_stability_null.py and renders a stand-alone       ║
# ║  preview plot: median curve + cross-seed IQR fill on a           ║
# ║  shared linear-y / log-instability axis.                         ║
# ║                                                                  ║
# ║  Used as a sanity check before wiring the band onto Panel B of   ║
# ║  the four-reduction internal-metrics figure - the dashed         ║
# ║  curve here is the d-ceiling reviewers will see overlaid on      ║
# ║  the real-data stability curves.                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Standalone preview of the R3 stability-null d-ceiling band."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from stag.constants import (
    RESULTS_DIR_DEFAULT,
    WONG,
    apply_figure_defaults,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--band-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "figures"
                / "figure2_stability_null_uniform.csv",
        help="Output of scripts/merge_stability_null.py.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULTS_DIR_DEFAULT / "figures",
    )
    parser.add_argument(
        "--stem", type=str, default="stability_null_band_preview",
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(6.0, 3.5),
    )
    parser.add_argument("--chosen-k", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.band_csv.exists():
        raise SystemExit(
            f"Band CSV not found at {args.band_csv}; "
            "run scripts/merge_stability_null.py first.",
        )

    band = pd.read_csv(args.band_csv).sort_values("k")
    print(f"Loaded {len(band)} k-rows from {args.band_csv}:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(band.to_string(index=False))

    apply_figure_defaults()
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    colour = WONG["black"]
    ax.fill_between(
        band["k"], band["instability_null_low"], band["instability_null_high"],
        facecolor="#888888", alpha=0.25, linewidth=0,
        label="IQR across 10 surrogate seeds",
    )
    ax.plot(
        band["k"], band["instability_null"],
        linestyle="--", marker="o", markersize=4,
        color=colour, linewidth=1.4, markeredgecolor="black",
        markeredgewidth=0.4, label="Median (d-ceiling)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("Hungarian-matched centroid drift  (null surrogate)")
    ax.set_title(
        "R3 stability null — uniform MaxAbs-box reference "
        "(10 seeds × 20 fits / k)",
        fontsize=10,
    )
    ax.axvline(args.chosen_k, color=WONG["vermilion"], linestyle=":",
               linewidth=1.0, alpha=0.7, label=f"chosen k = {args.chosen_k}")
    ax.legend(fontsize="x-small", loc="best", frameon=False)
    ax.set_xticks(band["k"].tolist())
    ax.tick_params(axis="x", labelsize="x-small")
    ax.grid(axis="y", which="both", linestyle=":", linewidth=0.4,
            color="#888888", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, args.stem, args.output_dir, data=band)
    print(
        f"\nWrote {args.output_dir / args.stem}.{{svg,png,csv}}",
    )


if __name__ == "__main__":
    main()
