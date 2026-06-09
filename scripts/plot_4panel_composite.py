#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.plot_4panel_composite                            ║
# ║  « 2 x 2 internal-metrics composite — silhouette / inertia /     ║
# ║    CH / stability — with the R3 null overlays where available »  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Layout (reading order):                                         ║
# ║    (A) Silhouette          (B) Inertia / Kneedle                 ║
# ║    (C) Quality vs. null    (D) Stability vs. null                ║
# ║                                                                  ║
# ║  All four panels share the x-axis (sharex=True) so the k-grid    ║
# ║  lines up exactly across rows.  Top row gets no x-labels (saved  ║
# ║  for the bottom row).                                            ║
# ║                                                                  ║
# ║  Style for every panel matches the standalone scripts:           ║
# ║    - sequential pink → deep purple delSize palette,              ║
# ║    - o ^ v < markers per delSize, IQR fills,                     ║
# ║    - electric blue-violet (#4903FC) dashed line + square         ║
# ║      markers for any null overlay,                               ║
# ║    - vermilion chosen-k = 8 line dotted across all panels.       ║
# ║                                                                  ║
# ║  Y-axes differ per panel:                                        ║
# ║    A linear   B log     C linear (1e7)    D asinh (in g)         ║
# ║                                                                  ║
# ║  Null overlays auto-detected from the band CSV column set, so    ║
# ║  this script gains panels A + B null bands the moment the next   ║
# ║  Aoraki re-run lands silhouette_null and inertia_null in         ║
# ║  figure2_stability_null_uniform.csv - no script edit needed.     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""2 x 2 internal-metrics composite — silhouette + inertia + CH + stability."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from matplotlib.ticker import NullFormatter, ScalarFormatter

from stag.constants import (
    MAXABS_SCALER_CSV,
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
        "--unscale-csv", type=Path, default=MAXABS_SCALER_CSV,
        help="MaxAbs divisors for converting the Panel D instability "
             "values to g-force at render time.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULTS_DIR_DEFAULT / "figures",
    )
    parser.add_argument(
        "--stem", type=str,
        default="internal_metrics_4panel_with_null",
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(9.0, 7.2),
        metavar=("WIDTH", "HEIGHT"),
        help="Matches the standalone four-panel internal-metrics figure.",
    )
    parser.add_argument("--chosen-k", type=int, default=8)
    parser.add_argument(
        "--asinh-linear-width", type=float, default=0.01,
        help="Panel D asinh y linear-region half-width in g.",
    )
    return parser.parse_args()


def _mean_divisor(unscale_csv: Path) -> float:
    divs = pd.read_csv(unscale_csv).iloc[0].to_numpy(dtype=np.float64)
    spread = (divs.max() - divs.min()) / divs.mean()
    if spread > 0.05:
        raise SystemExit(
            f"MaxAbs divisors {divs.tolist()} are non-uniform "
            f"(relative spread {spread:.3f}); cannot rescale CSV values "
            "by a scalar - re-run hungarian_centroid_drift with "
            "unscale=divs from centroids.",
        )
    return float(divs.mean())


def _kneedle_elbows(inertia_df: pd.DataFrame) -> dict[int, int]:
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


def _plot_delsize_traces(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    low_col: str | None,
    high_col: str | None,
) -> None:
    """Common pattern: one trace per delSize with IQR fill + markers."""
    for ds in sorted(df["delSize"].unique()):
        sub = df[df["delSize"] == ds].dropna(subset=[value_col]).sort_values("k")
        if sub.empty:
            continue
        colour = DELSIZE_PALETTE.get(int(ds), "#444444")
        marker = DELSIZE_MARKERS.get(int(ds), "o")
        line_w = 1.6 if int(ds) >= 25 else 1.3
        if low_col and high_col and {low_col, high_col}.issubset(sub.columns):
            ax.fill_between(
                sub["k"], sub[low_col], sub[high_col],
                facecolor=colour, alpha=0.25, linewidth=0,
            )
        ax.plot(
            sub["k"], sub[value_col],
            linestyle="-", marker=marker, markersize=4,
            color=colour, markeredgecolor="black", markeredgewidth=0.4,
            linewidth=line_w, label=f"delSize {ds} %",
        )


def _plot_null_trace(
    ax: plt.Axes,
    null_df: pd.DataFrame,
    value_col: str,
    low_col: str | None,
    high_col: str | None,
    label: str = "Uniform null (R3)",
) -> bool:
    """Overlay the null band on `ax` if the column is present.  Returns True iff drawn."""
    if value_col not in null_df.columns:
        return False
    sub = null_df.dropna(subset=[value_col]).sort_values("k")
    if sub.empty:
        return False
    if low_col and high_col and {low_col, high_col}.issubset(sub.columns):
        ax.fill_between(
            sub["k"], sub[low_col], sub[high_col],
            facecolor=NULL_COLOUR, alpha=0.18, linewidth=0,
        )
    ax.plot(
        sub["k"], sub[value_col],
        linestyle="--", marker=NULL_MARKER, markersize=4,
        color=NULL_COLOUR, markeredgecolor="black", markeredgewidth=0.4,
        linewidth=1.4, label=label,
    )
    return True


def main() -> None:
    args = parse_args()

    sil_df = pd.read_csv(args.silhouette_csv)
    real_df = pd.read_csv(args.real_csv)
    null_df = (
        pd.read_csv(args.null_csv) if args.null_csv.exists()
        else pd.DataFrame()
    )
    scale_g = _mean_divisor(args.unscale_csv)

    apply_figure_defaults()
    # Layout — reading order ABCD:
    #   (A) Silhouette    (B) CH with null
    #   (C) Inertia       (D) Stability with null
    fig, ((ax_sil, ax_ch), (ax_W, ax_inst)) = plt.subplots(
        2, 2, figsize=tuple(args.figsize), sharex=True,
    )

    # ── (A) Silhouette ───────────────────────────────────────────────
    _plot_delsize_traces(
        ax_sil, sil_df, "silhouette", "silhouette_low", "silhouette_high",
    )
    has_sil_null = (
        _plot_null_trace(
            ax_sil, null_df, "silhouette_null",
            "silhouette_null_low", "silhouette_null_high",
        )
        if not null_df.empty else False
    )
    ax_sil.set_ylabel(r"Mean silhouette ($\bar{s}$)")
    ax_sil.legend(fontsize="x-small", loc="best", frameon=False)

    # ── (B) Inertia / Kneedle ────────────────────────────────────────
    _plot_delsize_traces(
        ax_W, real_df, "inertia", "inertia_low", "inertia_high",
    )
    elbows = _kneedle_elbows(real_df)
    for ds, k_e in elbows.items():
        sub = real_df[(real_df["delSize"] == ds) & (real_df["k"] == k_e)]
        if not sub.empty:
            y_e = float(sub["inertia"].iloc[0])
            ax_W.scatter(
                [k_e], [y_e], s=80, facecolor="white",
                edgecolor=DELSIZE_PALETTE.get(int(ds), "#444444"),
                linewidth=2.0, zorder=6,
                label=f"Kneedle k={k_e} (delSize {int(ds)}%)",
            )
    has_inertia_null = (
        _plot_null_trace(
            ax_W, null_df, "inertia_null",
            "inertia_null_low", "inertia_null_high",
        )
        if not null_df.empty else False
    )
    ax_W.set_yscale("log")
    ax_W.set_ylabel(r"Inertia $W(k)$")
    ax_W.legend(fontsize="x-small", loc="best", frameon=False)

    # ── (C) CH index with null ───────────────────────────────────────
    _plot_delsize_traces(ax_ch, real_df, "ch", "ch_low", "ch_high")
    has_ch_null = (
        _plot_null_trace(
            ax_ch, null_df, "ch_null", "ch_null_low", "ch_null_high",
        )
        if not null_df.empty else False
    )
    ax_ch.set_ylabel("Calinski-Harabasz index")
    ax_ch.legend(fontsize="x-small", loc="best", frameon=False)
    ax_ch.yaxis.get_offset_text().set_fontsize(8)

    # ── (D) Stability with null (g-units, asinh y) ───────────────────
    real_g = real_df.copy()
    for col in ("instability", "instability_low", "instability_high"):
        if col in real_g.columns:
            real_g[f"{col}_g"] = real_g[col] * scale_g
    _plot_delsize_traces(
        ax_inst, real_g, "instability_g",
        "instability_low_g", "instability_high_g",
    )
    has_inst_null = False
    if not null_df.empty:
        null_g = null_df.copy()
        for col in ("instability_null",
                    "instability_null_low", "instability_null_high"):
            if col in null_g.columns:
                null_g[f"{col}_g"] = null_g[col] * scale_g
        has_inst_null = _plot_null_trace(
            ax_inst, null_g, "instability_null_g",
            "instability_null_low_g", "instability_null_high_g",
        )
    ax_inst.set_yscale("asinh", linear_width=args.asinh_linear_width)
    ax_inst.set_ylabel(r"Centroid drift (g)")
    ax_inst.legend(fontsize="x-small", loc="best", frameon=False)

    # ── Cosmetic + shared formatting ─────────────────────────────────
    for ax in (ax_sil, ax_W, ax_ch, ax_inst):
        ax.axvline(args.chosen_k, color=WONG["vermilion"], linestyle=":",
                   linewidth=1.0, alpha=0.7)
        ax.grid(axis="y", which="both", linestyle=":", linewidth=0.4,
                color="#888888", alpha=0.5)
        ax.set_axisbelow(True)
    for ax in (ax_ch, ax_inst):
        ax.set_xlabel("k-number")
    # sharex propagates ticks upward; set once on the bottom row.
    ax_inst.set_xticks([2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50])
    ax_inst.set_xlim(1.5, 51.5)
    ax_inst.tick_params(axis="x", labelsize="x-small")

    fig.tight_layout()

    combined = sil_df.merge(
        real_df, on=["delSize", "k"], how="outer", suffixes=("_sil", ""),
    )
    if not null_df.empty:
        combined = combined.merge(null_df, on="k", how="outer")
    save_figure(fig, args.stem, args.output_dir, data=combined)

    print(
        f"\nNull overlays: silhouette={has_sil_null}, inertia={has_inertia_null}, "
        f"ch={has_ch_null}, instability={has_inst_null}",
    )
    print(
        f"Wrote {args.output_dir / args.stem}.{{svg,png,csv}}  "
        f"(figsize = {args.figsize[0]} × {args.figsize[1]} in, sharex = True)",
    )


if __name__ == "__main__":
    main()
