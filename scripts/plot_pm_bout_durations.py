#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.plot_pm_bout_durations                           ║
# ║  « per-PM bout-duration table-figure (per-animal-first) »        ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Renders one row per prototypical movement (PM 0 .. PM 7) with   ║
# ║  the per-bout duration shown as a coloured IQR bar with the      ║
# ║  median as a dot.  Bout count printed on the right reads like a  ║
# ║  table cell.                                                     ║
# ║                                                                  ║
# ║  ANIMAL IS THE UNIT OF OBSERVATION.  For every PM we first       ║
# ║  compute that animal's median (and mean) bout duration, then     ║
# ║  aggregate across the 24 animals.  Pooling all bouts before      ║
# ║  computing moments would let the animal with the most bouts      ║
# ║  dominate the SEM / IQR — pseudo-replication.  We emit both:     ║
# ║                                                                  ║
# ║    - median of per-animal medians + IQR (figure default,         ║
# ║      robust to bout-duration right-skew)                         ║
# ║    - mean of per-animal means + SEM (manuscript-compatible       ║
# ║      style; SEM divided by sqrt(n_animals = 24), not sqrt(N))    ║
# ║                                                                  ║
# ║  Outputs:                                                        ║
# ║    pm_bout_durations_median_iqr.svg/png  — figure                ║
# ║    pm_bout_durations_median_iqr.csv      — aggregate table       ║
# ║    pm_bout_durations_per_animal.csv      — per-(animal, PM) rows ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Per-PM bout-duration table-figure (per-animal-first)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter

from stag.analysis.super_prototypes import (
    aggregate_durations_across_animals,
    per_animal_bout_streams,
    per_animal_pm_duration_stats,
)
from stag.constants import (
    CANONICAL_K8_LABELS,
    FPS,
    LABEL_TIMELINE_DEER_IDS,
    PM_COLOURS,
    PM_DISPLAY_NAMES,
    RESULTS_DIR_DEFAULT,
    apply_figure_defaults,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULTS_DIR_DEFAULT / "sprint3" / "figures",
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(7.0, 3.5),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default 7 × 3.5 to match the "
             "publication composite figure's row dimensions).",
    )
    parser.add_argument(
        "--stem", type=str, default="pm_bout_durations_median_iqr",
        help="Output filename stem (no extension).",
    )
    return parser.parse_args()


def _load_streams() -> dict[int, "object"]:
    print(f"Loading labels from {CANONICAL_K8_LABELS} ...")
    idx = np.asarray(np.load(CANONICAL_K8_LABELS, mmap_mode="r"))
    deer_ids = np.asarray(np.load(LABEL_TIMELINE_DEER_IDS, mmap_mode="r"))
    print(f"  idx shape = {idx.shape}, deer_ids shape = {deer_ids.shape}")
    streams = per_animal_bout_streams(idx, deer_ids)
    print(f"  {len(streams)} per-animal bout streams")
    return streams


def _render_figure(
    aggregate: pd.DataFrame,
    figsize: tuple[float, float],
) -> plt.Figure:
    """Median of per-animal medians + IQR across animals, one row per PM."""
    apply_figure_defaults()
    fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(len(aggregate))[::-1]

    for y, row in zip(y_positions, aggregate.itertuples()):
        colour = PM_COLOURS.get(int(row.pm), "#888888")
        ax.plot(
            [row.q25_of_medians_s, row.q75_of_medians_s], [y, y],
            color=colour, linewidth=5.5, alpha=0.5,
            solid_capstyle="round", zorder=2,
        )
        ax.plot(
            row.median_of_medians_s, y,
            marker="o", markersize=7, color=colour,
            markeredgecolor="black", markeredgewidth=0.5, zorder=3,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"PM{row.pm}  {PM_DISPLAY_NAMES.get(int(row.pm), row.pm)}"
         for row in aggregate.itertuples()],
        fontsize=9,
    )
    ax.set_ylim(-0.6, len(aggregate) - 0.4)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Bout duration (s)")
    ax.set_title(
        r"Per-PM bout duration  "
        r"$(\mathrm{median\ of\ per\text{-}animal\ medians} + "
        r"\mathrm{IQR\ across\ animals},\ n = 24)$",
        fontsize=10,
    )

    ax.grid(axis="x", which="both", linestyle=":", linewidth=0.4,
            color="#888888", alpha=0.6)
    ax.set_axisbelow(True)

    # Total-bout-count annotation to the right of each row.
    xmax = aggregate[["median_of_medians_s", "q75_of_medians_s"]].max().max()
    ann_x = xmax * 1.45
    for y, row in zip(y_positions, aggregate.itertuples()):
        ax.text(
            ann_x, y,
            f"n = {int(row.total_bouts):,}",
            ha="left", va="center", fontsize=8, color="#333333",
        )
    ax.set_xlim(right=ann_x * 2.5)

    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()

    streams = _load_streams()
    per_animal = per_animal_pm_duration_stats(streams, fps=FPS)
    aggregate = aggregate_durations_across_animals(per_animal)

    print("\nPer-animal table (head):")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(per_animal.head(10).to_string(index=False))

    print("\nCohort aggregate (per-animal-first, n = 24 animals):")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(aggregate.to_string(index=False))

    fig = _render_figure(aggregate, tuple(args.figsize))

    save_figure(fig, args.stem, args.output_dir, data=aggregate)
    per_animal_csv = args.output_dir / "pm_bout_durations_per_animal.csv"
    per_animal.to_csv(per_animal_csv, index=False)
    print(
        f"\nWrote {args.output_dir / args.stem}.{{svg,png,csv}}  "
        f"(figsize = {args.figsize[0]} × {args.figsize[1]} in)",
    )
    print(f"Wrote {per_animal_csv}")


if __name__ == "__main__":
    main()
