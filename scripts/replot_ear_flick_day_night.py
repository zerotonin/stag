#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.replot_ear_flick_day_night                       ║
# ║  « publication-ready paired day/night ear-flick rate »           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Re-renders results/sprint3/figures/ear_flick_day_night_paired   ║
# ║  for direct drop-in into the manuscript composite figure.        ║
# ║                                                                  ║
# ║  Layout tweaks vs the Sprint 3 default render:                   ║
# ║   - 2 : 1 aspect (twice as wide as tall) — fits the publication  ║
# ║     composite figure's lower row without further rescaling in    ║
# ║     Inkscape.                                                    ║
# ║   - semilog y-axis — rates span roughly 1e-3 to 3e-2 across      ║
# ║     animals, so a log axis keeps the lowest-rate animals off     ║
# ║     the zero baseline.  Plain-decimal tick labels via            ║
# ║     ScalarFormatter, not the matplotlib 10^{-n} default.         ║
# ║   - per-animal lines drawn in neutral grey, paired Day → Night   ║
# ║     boxplots in the canonical Wong day/night warm/cool pair      ║
# ║     (vermilion / blue).  Boxplots sit behind the lines so the    ║
# ║     individual-animal traces remain readable.                    ║
# ║   - Wilcoxon signed-rank result (W, p, median day/night ratio,   ║
# ║     n animals with day > night) annotated in the lower-right     ║
# ║     corner, recomputed from the CSV at render time so future     ║
# ║     re-runs cannot drift from the figure.                        ║
# ║                                                                  ║
# ║  Reads:  results/sprint3/figures/ear_flick_day_night_paired.csv  ║
# ║  Writes: same stem, .svg + .png + .csv via save_figure().        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Re-render the ear-flick day/night paired figure for publication."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.stats import wilcoxon

from stag.analysis.circadian import ear_flick_day_night_test
from stag.constants import (
    CANONICAL_K8_LABELS,
    LABEL_TIMELINE_DEER_IDS,
    LABEL_TIMELINE_TIMESTAMPS,
    PM_CATEGORY,
    RESULTS_DIR_DEFAULT,
    WONG,
    apply_figure_defaults,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "sprint3" / "figures"
                / "ear_flick_day_night_paired.csv",
        help="Per-animal CSV with rate_day / rate_night columns.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULTS_DIR_DEFAULT / "sprint3" / "figures",
        help="Destination for the re-rendered SVG + PNG + CSV.",
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(7.0, 3.5),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default 7 × 3.5, the 2 : 1 aspect "
             "the publication composite figure expects).",
    )
    parser.add_argument(
        "--recompute-from-labels", action="store_true",
        help="Re-run the day/night test from CANONICAL_K8_LABELS with the "
             "current _activity_pms denominator and overwrite --input-csv "
             "before plotting.  Required when the denominator definition "
             "changes (e.g. from non-quiescent to non-resting).",
    )
    return parser.parse_args()


def _recompute_csv(input_csv: Path) -> None:
    """Re-run the day/night test from labels and overwrite ``input_csv``.

    Denominator: every PM except PM 0 (Quiescent) and PM 1 (Resting) —
    i.e. "non-resting" samples.  Numerator: PM 2 + PM 4 + PM 5
    (the three ear-flick prototypes).
    """
    print(f"Loading labels from {CANONICAL_K8_LABELS} ...")
    idx = np.load(CANONICAL_K8_LABELS, mmap_mode="r")
    deer_ids = np.load(LABEL_TIMELINE_DEER_IDS, mmap_mode="r")
    timestamps_ns = np.load(LABEL_TIMELINE_TIMESTAMPS, mmap_mode="r")
    print(
        f"  idx shape = {idx.shape}, "
        f"deer_ids shape = {deer_ids.shape}, "
        f"timestamps shape = {timestamps_ns.shape}",
    )

    ear_pms = [pm for pm, cat in PM_CATEGORY.items() if cat == "ear_flick"]
    # Non-resting: every PM that is NOT quiescent (PM 0) and NOT resting (PM 1).
    activity_pms = [pm for pm in PM_CATEGORY if pm not in (0, 1)]
    print(f"  ear-flick PMs (numerator)  : {sorted(ear_pms)}")
    print(f"  activity PMs (denominator) : {sorted(activity_pms)}  (non-resting)")

    test = ear_flick_day_night_test(
        np.asarray(idx), np.asarray(timestamps_ns), np.asarray(deer_ids),
        ear_flick_pms=ear_pms, activity_pms=activity_pms,
    )
    out_df = test["per_animal"]
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(input_csv)
    print(
        f"Wrote {input_csv}  "
        f"(W = {test['W']}, p = {test['p_value']:.4g}, "
        f"median ratio = {test['median_ratio_day_over_night']:.3f})",
    )


def main() -> None:
    args = parse_args()
    if args.recompute_from_labels:
        _recompute_csv(args.input_csv)
    df = pd.read_csv(args.input_csv)
    if not {"rate_day", "rate_night"}.issubset(df.columns):
        raise SystemExit(
            f"{args.input_csv} missing rate_day / rate_night columns; "
            f"found {list(df.columns)}",
        )

    # ── Wilcoxon signed-rank (recomputed from the CSV) ────────────────
    test = wilcoxon(
        df["rate_day"].to_numpy(), df["rate_night"].to_numpy(),
        alternative="two-sided",
    )
    W = float(test.statistic)
    p_value = float(test.pvalue)
    # Derive day/night ratio per-animal so the script tolerates either CSV
    # schema (the test function emits rate_day/rate_night only; older
    # snapshots also carried a precomputed `ratio` column).
    if "ratio" in df.columns:
        ratio_series = df["ratio"]
    else:
        ratio_series = df["rate_day"] / df["rate_night"].replace(0, np.nan)
    median_ratio = float(ratio_series.median())
    n_animals = int(len(df))
    n_day_gt_night = int((df["rate_day"] > df["rate_night"]).sum())

    apply_figure_defaults()
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    x_day, x_night = 0.0, 1.0
    jitter = (np.random.default_rng(0).random(len(df)) - 0.5) * 0.06

    # ── Boxplots first (behind the lines) ─────────────────────────────
    box_kw = dict(
        widths=0.36, patch_artist=True, manage_ticks=False,
        showfliers=False, zorder=1,
    )
    box_day = ax.boxplot(df["rate_day"], positions=[x_day], **box_kw)
    box_night = ax.boxplot(df["rate_night"], positions=[x_night], **box_kw)
    for box_dict, fill in (
        (box_day, WONG["vermilion"]),
        (box_night, WONG["blue"]),
    ):
        for patch in box_dict["boxes"]:
            patch.set_facecolor(fill)
            patch.set_alpha(0.35)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        for line in box_dict["whiskers"] + box_dict["caps"]:
            line.set_color("black")
            line.set_linewidth(0.8)
        for line in box_dict["medians"]:
            line.set_color("black")
            line.set_linewidth(1.6)

    # ── Per-animal paired lines (neutral grey, on top of the boxes) ──
    line_colour = "#444444"
    for (_, row), j in zip(df.iterrows(), jitter):
        ax.plot(
            [x_day + j, x_night + j],
            [row["rate_day"], row["rate_night"]],
            color=line_colour, linewidth=0.8, alpha=0.35,
            marker="o", markersize=2.8, markerfacecolor=line_colour,
            markeredgecolor="black", markeredgewidth=0.3, zorder=2,
        )

    # ── Axes ──────────────────────────────────────────────────────────
    ax.set_yscale("log")
    # Plain decimal tick labels (0.001 / 0.01 / 0.03) rather than 10^{-n}.
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([x_day, x_night])
    ax.set_xticklabels(["Day", "Night"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel(
        r"Ear-flick rate $\left(\dfrac{\mathrm{ear\,flick}}{\mathrm{non\,resting}}\right)$",
    )
    ax.set_title("Per-animal ear-flick rate, day vs night")
    ax.grid(axis="y", which="both", linestyle=":", linewidth=0.4,
            color="#888888", alpha=0.6)
    ax.set_axisbelow(True)

    # ── Stats annotation in the lower-right ──────────────────────────
    stats_text = (
        f"Wilcoxon signed-rank\n"
        f"$W = {W:.1f}$,  $p = {p_value:.2g}$\n"
        f"Median day/night ratio = {median_ratio:.2f}\n"
        f"Day > night in {n_day_gt_night}/{n_animals} animals"
    )
    ax.text(
        0.98, 0.04, stats_text,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7,
        bbox=dict(
            facecolor="white", edgecolor="#bbbbbb",
            boxstyle="round,pad=0.32", linewidth=0.6,
        ),
    )

    fig.tight_layout()
    print(
        f"  Wilcoxon W = {W:.3f}, p = {p_value:.3g}, "
        f"median ratio = {median_ratio:.3f}, "
        f"day > night in {n_day_gt_night}/{n_animals} animals",
    )

    save_figure(
        fig, "ear_flick_day_night_paired", args.output_dir, data=df,
    )
    print(
        f"Wrote {args.output_dir / 'ear_flick_day_night_paired'}.{{svg,png,csv}}",
    )
    print(
        f"  figsize = {args.figsize[0]} × {args.figsize[1]} in,  yscale = log",
    )


if __name__ == "__main__":
    main()
