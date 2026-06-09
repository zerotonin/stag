#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.plot_silhouette_inertia_vertical                 ║
# ║  « Vertical 2 x 1 panel C + D, aligned x-axis »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Stacks the silhouette panel on top of the inertia / Kneedle     ║
# ║  panel with shared x-axis so they line up under the composite    ║
# ║  figure's top-row CH + stability pair.                           ║
# ║                                                                  ║
# ║  Style mirrors                                                   ║
# ║  ``plot_ch_panel_with_null`` and                                 ║
# ║  ``plot_instability_panel_with_null``:                           ║
# ║    - sequential pink → deep purple delSize palette,              ║
# ║    - o ^ v < markers, IQR fills,                                 ║
# ║    - matching x-tick set,                                        ║
# ║    - vermilion chosen-k line.                                    ║
# ║                                                                  ║
# ║  Currently real-data only.  The R3 null sweep is re-running on   ║
# ║  Aoraki with silhouette + inertia recording added; once the      ║
# ║  band CSV gains those columns this script grows two overlays in  ║
# ║  the same #4903FC-square style as the other panels.              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Vertical 2 x 1 silhouette + inertia / Kneedle panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from kneed import KneeLocator

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
        "--silhouette-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "tables"
                / "silhouette_per_delSize_k_extended.csv",
    )
    parser.add_argument(
        "--inertia-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "figures"
                / "figure2_internal_metrics_4reductions.csv",
    )
    parser.add_argument(
        "--null-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "figures"
                / "figure2_stability_null_uniform.csv",
        help="Null band CSV.  Used iff it has silhouette_null and "
             "inertia_null columns - otherwise the script silently "
             "skips the null overlays.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULTS_DIR_DEFAULT / "figures",
    )
    parser.add_argument(
        "--stem", type=str,
        default="silhouette_inertia_vertical",
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(4.5, 7.2),
        metavar=("WIDTH", "HEIGHT"),
        help="Match the right-hand column of a 9 x 7.2 in 2x2 "
             "composite (i.e. one column, two stacked panels).",
    )
    parser.add_argument("--chosen-k", type=int, default=8)
    return parser.parse_args()


def _kneedle_elbows(inertia_df: pd.DataFrame) -> dict[int, int]:
    """Per-delSize Kneedle elbow on the median inertia curve."""
    elbows: dict[int, int] = {}
    for ds, sub in inertia_df.groupby("delSize"):
        sub = sub.sort_values("k").dropna(subset=["inertia"])
        if len(sub) < 3:
            continue
        try:
            knee = KneeLocator(
                sub["k"], sub["inertia"],
                curve="convex", direction="decreasing", S=1.0,
            )
        except Exception:
            continue
        if knee.knee is not None:
            elbows[int(ds)] = int(knee.knee)
    return elbows


def _has_null_columns(null_df: pd.DataFrame) -> tuple[bool, bool]:
    return (
        "silhouette_null" in null_df.columns,
        "inertia_null" in null_df.columns,
    )


def main() -> None:
    args = parse_args()

    sil_df = pd.read_csv(args.silhouette_csv)
    inertia_df = pd.read_csv(args.inertia_csv)
    null_df: pd.DataFrame | None = None
    if args.null_csv.exists():
        null_df = pd.read_csv(args.null_csv)
    has_sil_null, has_inertia_null = (
        (False, False) if null_df is None else _has_null_columns(null_df)
    )
    print(
        f"Null overlays: silhouette = {has_sil_null}, "
        f"inertia = {has_inertia_null}",
    )

    apply_figure_defaults()
    fig, (ax_sil, ax_W) = plt.subplots(
        2, 1, figsize=tuple(args.figsize), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    # ── Panel C: silhouette ───────────────────────────────────────────
    for ds in sorted(sil_df["delSize"].unique()):
        sub = sil_df[sil_df["delSize"] == ds].sort_values("k")
        if sub.empty:
            continue
        colour = DELSIZE_PALETTE.get(int(ds), "#444444")
        marker = DELSIZE_MARKERS.get(int(ds), "o")
        line_w = 1.6 if int(ds) >= 25 else 1.3
        if {"silhouette_low", "silhouette_high"}.issubset(sub.columns):
            ax_sil.fill_between(
                sub["k"], sub["silhouette_low"], sub["silhouette_high"],
                facecolor=colour, alpha=0.25, linewidth=0,
            )
        ax_sil.plot(
            sub["k"], sub["silhouette"],
            linestyle="-", marker=marker, markersize=4,
            color=colour, markeredgecolor="black", markeredgewidth=0.4,
            linewidth=line_w, label=f"delSize {ds} %",
        )
    if has_sil_null:
        nfin = null_df.dropna(subset=["silhouette_null"]).sort_values("k")
        if {"silhouette_null_low", "silhouette_null_high"}.issubset(
                nfin.columns):
            ax_sil.fill_between(
                nfin["k"],
                nfin["silhouette_null_low"], nfin["silhouette_null_high"],
                facecolor=NULL_COLOUR, alpha=0.18, linewidth=0,
            )
        ax_sil.plot(
            nfin["k"], nfin["silhouette_null"],
            linestyle="--", marker=NULL_MARKER, markersize=4,
            color=NULL_COLOUR, markeredgecolor="black", markeredgewidth=0.4,
            linewidth=1.4, label="Uniform null (R3)",
        )
    ax_sil.set_ylabel(r"Mean silhouette ($\bar{s}$)")
    ax_sil.set_title("(A) Silhouette", fontsize=10)
    ax_sil.axvline(args.chosen_k, color=WONG["vermilion"], linestyle=":",
                   linewidth=1.0, alpha=0.7)
    ax_sil.legend(fontsize="x-small", loc="best", frameon=False)
    ax_sil.grid(axis="y", which="both", linestyle=":", linewidth=0.4,
                color="#888888", alpha=0.5)
    ax_sil.set_axisbelow(True)

    # ── Panel D: inertia + Kneedle elbows ─────────────────────────────
    elbows = _kneedle_elbows(inertia_df)
    for ds in sorted(inertia_df["delSize"].unique()):
        sub = inertia_df[inertia_df["delSize"] == ds].dropna(
            subset=["inertia"]).sort_values("k")
        if sub.empty:
            continue
        colour = DELSIZE_PALETTE.get(int(ds), "#444444")
        marker = DELSIZE_MARKERS.get(int(ds), "o")
        line_w = 1.6 if int(ds) >= 25 else 1.3
        ax_W.fill_between(
            sub["k"], sub["inertia_low"], sub["inertia_high"],
            facecolor=colour, alpha=0.25, linewidth=0,
        )
        ax_W.plot(
            sub["k"], sub["inertia"],
            linestyle="-", marker=marker, markersize=4,
            color=colour, markeredgecolor="black", markeredgewidth=0.4,
            linewidth=line_w, label=f"delSize {ds} %",
        )
        if int(ds) in elbows:
            k_e = elbows[int(ds)]
            row_e = sub[sub["k"] == k_e]
            if not row_e.empty:
                y_e = float(row_e["inertia"].iloc[0])
                ax_W.scatter(
                    [k_e], [y_e], s=80, facecolor="white",
                    edgecolor=colour, linewidth=2.0, zorder=6,
                    label=f"Kneedle k = {k_e} (delSize {int(ds)} %)",
                )
    if has_inertia_null:
        nfin = null_df.dropna(subset=["inertia_null"]).sort_values("k")
        if {"inertia_null_low", "inertia_null_high"}.issubset(nfin.columns):
            ax_W.fill_between(
                nfin["k"],
                nfin["inertia_null_low"], nfin["inertia_null_high"],
                facecolor=NULL_COLOUR, alpha=0.18, linewidth=0,
            )
        ax_W.plot(
            nfin["k"], nfin["inertia_null"],
            linestyle="--", marker=NULL_MARKER, markersize=4,
            color=NULL_COLOUR, markeredgecolor="black", markeredgewidth=0.4,
            linewidth=1.4, label="Uniform null (R3)",
        )
    ax_W.set_yscale("log")
    ax_W.set_xlabel("k-number")
    ax_W.set_ylabel(r"Inertia $W(k)$")
    ax_W.set_title("(B) Inertia / Kneedle elbow", fontsize=10)
    ax_W.axvline(args.chosen_k, color=WONG["vermilion"], linestyle=":",
                 linewidth=1.0, alpha=0.7)
    ax_W.legend(fontsize="x-small", loc="best", frameon=False)
    ax_W.grid(axis="y", which="both", linestyle=":", linewidth=0.4,
              color="#888888", alpha=0.5)
    ax_W.set_axisbelow(True)

    # ── Shared x-axis (set on bottom panel — sharex propagates up) ────
    ax_W.set_xticks([2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50])
    ax_W.set_xlim(1.5, 51.5)
    ax_W.tick_params(axis="x", labelsize="x-small")

    fig.tight_layout()

    combined = sil_df.merge(
        inertia_df[["delSize", "k", "inertia", "inertia_low", "inertia_high"]],
        on=["delSize", "k"], how="outer",
    )
    save_figure(fig, args.stem, args.output_dir, data=combined)
    print(
        f"\nWrote {args.output_dir / args.stem}.{{svg,png,csv}}  "
        f"(figsize = {args.figsize[0]} × {args.figsize[1]} in, "
        f"sharex = True)",
    )


if __name__ == "__main__":
    main()
