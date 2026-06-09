#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.plot_ch_panel_with_null                          ║
# ║  « Panel A drop-in: Calinski-Harabasz + R3 null d-floor »        ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Single-panel render of the four delSize real-data CH curves     ║
# ║  plus the R3 uniform-MaxAbs-box null d-floor, sized for the      ║
# ║  Panel A slot of the composite internal-metrics figure.          ║
# ║                                                                  ║
# ║  Style mirrors                                                   ║
# ║  ``scripts.plot_instability_panel_with_null``:                   ║
# ║    - sequential pink → deep purple palette for the four          ║
# ║      delSize traces (#EECCC9, #DB97AA, #A75491, #291E38),        ║
# ║      o ^ v < markers, IQR fills,                                 ║
# ║    - electric blue-violet ``#4903FC`` dashed line with square    ║
# ║      markers for the null,                                       ║
# ║    - x-ticks matching the composite,                             ║
# ║    - **linear y-axis** (CH is always positive and the real-data  ║
# ║      curve dominates the chart - the null compresses to the      ║
# ║      baseline, which is precisely the visual claim: real CH is   ║
# ║      orders of magnitude above what a structureless reference    ║
# ║      can muster).                                                ║
# ║                                                                  ║
# ║  Reads:                                                          ║
# ║    results/figures/figure2_internal_metrics_4reductions.csv      ║
# ║    results/figures/figure2_stability_null_uniform.csv (CH band   ║
# ║      added by scripts/merge_stability_null.py).                  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Single-panel CH + null d-floor, drop-in for the composite figure."""

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


DELSIZE_PALETTE: dict[int, str] = {
    0:  "#EECCC9",
    10: "#DB97AA",
    25: "#A75491",
    50: "#291E38",
}

DELSIZE_MARKERS: dict[int, str] = {
    0:  "o",
    10: "^",
    25: "v",
    50: "<",
}

NULL_COLOUR: str = "#4903FC"
NULL_MARKER: str = "s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "figures"
                / "figure2_internal_metrics_4reductions.csv",
    )
    parser.add_argument(
        "--null-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "figures"
                / "figure2_stability_null_uniform.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULTS_DIR_DEFAULT / "figures",
    )
    parser.add_argument(
        "--stem", type=str,
        default="ch_panel_with_null",
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(4.5, 3.6),
        metavar=("WIDTH", "HEIGHT"),
        help="Match the subplot A slot in the 9 × 7.2 in four-panel "
             "figure (4.5 × 3.6 in per cell).",
    )
    parser.add_argument("--chosen-k", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    real = pd.read_csv(args.real_csv).sort_values(["delSize", "k"])
    null = pd.read_csv(args.null_csv).sort_values("k")

    if "ch_null" not in null.columns:
        raise SystemExit(
            f"{args.null_csv} does not carry ch_null columns; "
            "re-run scripts/merge_stability_null.py to regenerate it.",
        )

    apply_figure_defaults()
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    # ── Real-data delSize traces ──────────────────────────────────────
    for ds in sorted(real["delSize"].unique()):
        sub = real[real["delSize"] == ds].dropna(subset=["ch"])
        if sub.empty:
            continue
        colour = DELSIZE_PALETTE.get(int(ds), "#444444")
        marker = DELSIZE_MARKERS.get(int(ds), "o")
        line_w = 1.6 if int(ds) >= 25 else 1.3
        ax.fill_between(
            sub["k"], sub["ch_low"], sub["ch_high"],
            facecolor=colour, alpha=0.25, linewidth=0,
        )
        ax.plot(
            sub["k"], sub["ch"],
            linestyle="-", marker=marker, markersize=4,
            color=colour, markeredgecolor="black", markeredgewidth=0.4,
            linewidth=line_w, label=f"delSize {ds} %",
        )

    # ── Null d-floor ──────────────────────────────────────────────────
    null_finite = null.dropna(subset=["ch_null"]).sort_values("k")
    ax.fill_between(
        null_finite["k"],
        null_finite["ch_null_low"],
        null_finite["ch_null_high"],
        facecolor=NULL_COLOUR, alpha=0.18, linewidth=0,
    )
    ax.plot(
        null_finite["k"], null_finite["ch_null"],
        linestyle="--", marker=NULL_MARKER, markersize=4,
        color=NULL_COLOUR, markeredgecolor="black", markeredgewidth=0.4,
        linewidth=1.4, label="Uniform null (R3 d-floor)",
    )

    # ── Axes ──────────────────────────────────────────────────────────
    # CH > 0 and the real-data curve dominates by ~3 orders of
    # magnitude over the null — linear y keeps that scale honest and
    # lets a reviewer read "the null is essentially the baseline".
    ax.set_xlabel("k-number")
    ax.set_ylabel("Calinski–Harabasz index")
    ax.axvline(args.chosen_k, color=WONG["vermilion"], linestyle=":",
               linewidth=1.0, alpha=0.7)
    ax.set_xticks([2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xlim(1.5, 51.5)
    ax.tick_params(axis="x", labelsize="x-small")
    ax.legend(fontsize="x-small", loc="best", frameon=False)
    ax.grid(axis="y", which="both", linestyle=":", linewidth=0.4,
            color="#888888", alpha=0.5)
    ax.set_axisbelow(True)

    # Matplotlib's auto offset_text label ("1e7") sits in the corner;
    # nudge font size down to match the other tick labels.
    ax.yaxis.get_offset_text().set_fontsize(8)

    fig.tight_layout()

    combined = real.merge(
        null[["k", "ch_null", "ch_null_low", "ch_null_high"]],
        on="k", how="outer",
    )
    save_figure(fig, args.stem, args.output_dir, data=combined)
    print(
        f"\nWrote {args.output_dir / args.stem}.{{svg,png,csv}}  "
        f"(figsize = {args.figsize[0]} × {args.figsize[1]} in, "
        f"yscale = linear)",
    )


if __name__ == "__main__":
    main()
