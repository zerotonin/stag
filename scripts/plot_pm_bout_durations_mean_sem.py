#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.plot_pm_bout_durations_mean_sem                  ║
# ║  « horizontal-bar per-PM duration panel, mean ± SEM »            ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Mirrors panel (c) of the manuscript's Figure 3 family heat-     ║
# ║  map composite: one horizontal bar per prototypical movement     ║
# ║  with the mean bout duration as length and the per-animal SEM    ║
# ║  as a horizontal error-bar.  The mean value is printed in        ║
# ║  X.XX s notation next to each bar so the panel reads like a      ║
# ║  table.                                                          ║
# ║                                                                  ║
# ║  PER-ANIMAL-FIRST aggregation: for each PM we compute the 24     ║
# ║  per-animal means, then report mean-of-means ± SEM(24).  This    ║
# ║  honestly reports cohort variability (the manuscript's existing  ║
# ║  centroid_label_info.json pools all bouts before computing the   ║
# ║  SEM and so under-states its denominator by ~6 orders of         ║
# ║  magnitude).                                                     ║
# ║                                                                  ║
# ║  Rows are ordered by behavioural family — Inactive (PM0 / PM1 /  ║
# ║  PM3) on top, Grazing (PM6 / PM7) in the middle, Ear flicks      ║
# ║  (PM2 / PM4 / PM5) at the bottom — with the family colour band   ║
# ║  drawn behind each group and labelled in the left margin.        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Horizontal-bar per-PM duration panel, mean ± SEM (per-animal-first)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from stag.analysis.super_prototypes import (
    aggregate_durations_across_animals,
    per_animal_bout_streams,
    per_animal_pm_duration_stats,
)
from stag.constants import (
    CANONICAL_K8_LABELS,
    FPS,
    LABEL_TIMELINE_DEER_IDS,
    PM_CATEGORY,
    PM_CATEGORY_COLOURS,
    PM_DISPLAY_NAMES,
    RESULTS_DIR_DEFAULT,
    WONG,
    apply_figure_defaults,
    save_figure,
)

# Family ordering of PMs — matches panel (c) of the heatmap composite:
#   top    Inactive   PM 0, PM 1, PM 3
#   mid    Grazing    PM 6, PM 7
#   bottom Ear flicks PM 2, PM 4, PM 5
FAMILY_ORDER: list[tuple[str, list[int]]] = [
    ("inactive",  [0, 1, 3]),
    ("grazing",   [6, 7]),
    ("ear_flick", [2, 4, 5]),
]

FAMILY_DISPLAY: dict[str, str] = {
    "inactive":  "Inactive",
    "grazing":   "Grazing",
    "ear_flick": "Ear flicks",
}

BAR_FILL: str = WONG["orange"]      # warm orange, matches panel (c)
BAR_EDGE: str = "#7a4a00"
ERR_COLOUR: str = "#333333"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULTS_DIR_DEFAULT / "sprint3" / "figures",
    )
    parser.add_argument(
        "--stem", type=str, default="pm_bout_durations_mean_sem",
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(4.5, 4.8),
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument(
        "--xmax", type=float, default=1.0,
        help="Upper limit of the duration x-axis (seconds).",
    )
    return parser.parse_args()


def _load_streams() -> dict:
    print(f"Loading labels from {CANONICAL_K8_LABELS} ...")
    idx = np.asarray(np.load(CANONICAL_K8_LABELS, mmap_mode="r"))
    deer_ids = np.asarray(np.load(LABEL_TIMELINE_DEER_IDS, mmap_mode="r"))
    print(f"  idx shape = {idx.shape}, deer_ids shape = {deer_ids.shape}")
    streams = per_animal_bout_streams(idx, deer_ids)
    print(f"  {len(streams)} per-animal bout streams")
    return streams


def _ordered_pm_rows(aggregate: pd.DataFrame) -> list[dict]:
    """Return one row per PM in FAMILY_ORDER, top → bottom in the figure."""
    rows: list[dict] = []
    for family, pm_list in FAMILY_ORDER:
        for pm in pm_list:
            sub = aggregate[aggregate["pm"] == pm]
            if sub.empty:
                continue
            rec = sub.iloc[0].to_dict()
            rec["family"] = family
            rec["family_display"] = FAMILY_DISPLAY[family]
            rec["family_colour"] = PM_CATEGORY_COLOURS[family]
            rec["display_name"] = PM_DISPLAY_NAMES.get(int(pm), str(pm))
            rows.append(rec)
    return rows


def _render(rows: list[dict], figsize: tuple[float, float], xmax: float) -> plt.Figure:
    apply_figure_defaults()
    fig, ax = plt.subplots(figsize=figsize)

    n = len(rows)
    # Rows top → bottom: assign descending y positions
    y_positions = np.arange(n)[::-1]

    # ── Background family bands ──────────────────────────────────────
    fam_block_start: int | None = None
    last_fam: str | None = None
    for y, row in zip(y_positions, rows):
        if last_fam is None:
            last_fam, fam_block_start = row["family"], y
        elif row["family"] != last_fam:
            ax.axhspan(
                fam_block_start + 0.5, y + 0.5,
                facecolor=PM_CATEGORY_COLOURS[last_fam], alpha=0.10, zorder=0,
            )
            last_fam, fam_block_start = row["family"], y
    if last_fam is not None and fam_block_start is not None:
        ax.axhspan(
            fam_block_start + 0.5, y_positions[-1] - 0.5,
            facecolor=PM_CATEGORY_COLOURS[last_fam], alpha=0.10, zorder=0,
        )

    # ── Bars + SEM error bars + value annotation ─────────────────────
    for y, row in zip(y_positions, rows):
        mean = float(row["mean_of_means_s"])
        sem = float(row["sem_s"]) if not pd.isna(row["sem_s"]) else 0.0
        ax.barh(
            y, mean, height=0.65,
            color=BAR_FILL, edgecolor=BAR_EDGE, linewidth=0.6, zorder=2,
        )
        ax.errorbar(
            mean, y, xerr=sem,
            ecolor=ERR_COLOUR, elinewidth=0.8, capsize=2.5,
            capthick=0.8, zorder=3,
        )
        # Value label to the right of the bar (just past the error-bar cap)
        ax.text(
            mean + sem + xmax * 0.018, y,
            f"{mean:.2f} s",
            ha="left", va="center", fontsize=8, color="#222222",
        )

    # ── Y axis labels (PM display names with PM index) ───────────────
    labels = [f"PM{int(row['pm'])}: {row['display_name']}" for row in rows]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.6, n - 0.4)

    # ── Family colour sidebar + label on the left ────────────────────
    # x position in axes coords (negative = outside the data area).
    fam_groups: dict[str, list[int]] = {}
    for y, row in zip(y_positions, rows):
        fam_groups.setdefault(row["family"], []).append(int(y))
    for family, ys in fam_groups.items():
        top, bottom = max(ys), min(ys)
        # Coloured vertical rectangle just outside the y-axis tick labels.
        ax.add_patch(Rectangle(
            xy=(-xmax * 0.45, bottom - 0.4),
            width=xmax * 0.018,
            height=top - bottom + 0.8,
            transform=ax.transData,
            facecolor=PM_CATEGORY_COLOURS[family],
            edgecolor="none",
            clip_on=False, zorder=1,
        ))
        # Rotated family-name label inside the coloured strip.
        ax.text(
            -xmax * 0.475, (top + bottom) / 2,
            FAMILY_DISPLAY[family],
            ha="center", va="center", rotation=90,
            fontsize=9, fontweight="bold",
            color=PM_CATEGORY_COLOURS[family],
        )

    # ── Axes formatting ──────────────────────────────────────────────
    ax.set_xlim(0, xmax)
    ax.set_xlabel(r"seconds (mean $\pm$ SEM)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", which="major", linestyle=":", linewidth=0.4,
            color="#888888", alpha=0.5)
    ax.set_axisbelow(True)
    # Hide top + right spines for cleanness (matches panel (c) style)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    # Extra left margin for the rotated family labels
    fig.subplots_adjust(left=0.32)
    return fig


def main() -> None:
    args = parse_args()

    streams = _load_streams()
    per_animal = per_animal_pm_duration_stats(streams, fps=FPS)
    aggregate = aggregate_durations_across_animals(per_animal)
    print("\nPer-animal-first cohort aggregate:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(aggregate.to_string(index=False))

    rows = _ordered_pm_rows(aggregate)
    fig = _render(rows, tuple(args.figsize), args.xmax)

    # Companion CSV — same order as the chart rows
    chart_df = pd.DataFrame(rows)[[
        "pm", "display_name", "family",
        "n_animals", "mean_of_means_s", "sem_s", "total_bouts",
    ]]
    save_figure(fig, args.stem, args.output_dir, data=chart_df)
    print(
        f"\nWrote {args.output_dir / args.stem}.{{svg,png,csv}}  "
        f"(figsize = {args.figsize[0]} × {args.figsize[1]} in)",
    )


if __name__ == "__main__":
    main()
