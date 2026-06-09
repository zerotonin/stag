#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.plot_instability_panel_with_null                 ║
# ║  « Panel D drop-in: real-data stability + R3 null d-ceiling »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Single-panel render of the four delSize stability curves plus   ║
# ║  the R3 uniform-MaxAbs-box null d-ceiling, sized to match the    ║
# ║  subplot D slot of the existing four-reduction internal-metrics  ║
# ║  figure so it can be dropped straight into the publication       ║
# ║  composite with no rescaling in Inkscape.                        ║
# ║                                                                  ║
# ║  Units: the y-axis is in g-force (Earth-gravity).  Centroids in  ║
# ║  the input CSVs are stored in MaxAbs space; multiplying by the   ║
# ║  per-column divisors (~7.97 g for every dim, since col-5 clips   ║
# ║  at ±7.99 g and the other five dims are also accelerometer-      ║
# ║  ranged) yields the physical-units instability without re-       ║
# ║  running k-means.                                                ║
# ║                                                                  ║
# ║  Axis: asinh y — symmetric-log transition near zero so the very- ║
# ║  small real-data values stay visible alongside the order-of-     ║
# ║  magnitude-larger null band.                                     ║
# ║                                                                  ║
# ║  Style: existing delSize palette + markers for the four real     ║
# ║  curves; null overlay drawn in deep navy with square markers     ║
# ║  per the publication-figure convention.                          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Single-panel stability + d-ceiling, drop-in for the composite figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stag.constants import (
    MAXABS_SCALER_CSV,
    RESULTS_DIR_DEFAULT,
    WONG,
    apply_figure_defaults,
    save_figure,
)

# Sequential pink → deep purple palette, lab canonical for the
# leave-out-reduction overlays (matches the silhouette + elbow figure).
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

# Electric blue-violet for the null overlay - bright enough to read
# clearly against the dark deep-purple delSize 50 trace (#291E38)
# even at print size.
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
        "--unscale-csv", type=Path, default=MAXABS_SCALER_CSV,
        help="Per-column MaxAbs divisors.  Existing CSV values are in "
             "MaxAbs space; we multiply by these to convert to g-force "
             "without re-running k-means.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULTS_DIR_DEFAULT / "figures",
    )
    parser.add_argument(
        "--stem", type=str,
        default="instability_panel_with_null_grams",
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(4.5, 3.6),
        metavar=("WIDTH", "HEIGHT"),
        help="Match the subplot D slot in the 9 × 7.2 in four-panel "
             "figure (4.5 × 3.6 in per cell).",
    )
    parser.add_argument("--chosen-k", type=int, default=8)
    parser.add_argument(
        "--asinh-linear-width", type=float, default=0.01,
        help="Linear-region half-width of the asinh y-axis (in g).  "
             "0.01 g keeps low-real-data values visible.",
    )
    return parser.parse_args()


def _mean_divisor(unscale_csv: Path) -> float:
    """Single multiplicative factor for the (near-uniform) MaxAbs divisors."""
    divs = pd.read_csv(unscale_csv).iloc[0].to_numpy(dtype=np.float64)
    spread = (divs.max() - divs.min()) / divs.mean()
    if spread > 0.05:
        # Distances are no longer a single scalar transform; caller would
        # need to redo Hungarian-matching from the centroids in physical
        # space.  Refuse to silently apply a mean-divisor approximation.
        raise SystemExit(
            f"MaxAbs divisors {divs.tolist()} are non-uniform "
            f"(relative spread {spread:.3f}); cannot rescale CSV values "
            "by a scalar — re-run hungarian_centroid_drift / "
            "compute_instability with unscale=divs from centroids.",
        )
    return float(divs.mean())


def main() -> None:
    args = parse_args()

    scale = _mean_divisor(args.unscale_csv)
    print(f"Uniform per-column scaling factor: {scale:.4f} g")

    real = pd.read_csv(args.real_csv).sort_values(["delSize", "k"])
    null = pd.read_csv(args.null_csv).sort_values("k")

    # Multiply instability columns to lift them out of MaxAbs space.
    for col in ("instability", "instability_low", "instability_high"):
        if col in real.columns:
            real[f"{col}_g"] = real[col] * scale
    for col in ("instability_null", "instability_null_low",
                "instability_null_high"):
        if col in null.columns:
            null[f"{col}_g"] = null[col] * scale

    apply_figure_defaults()
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    # ── Real-data delSize traces ──────────────────────────────────────
    for ds in sorted(real["delSize"].unique()):
        sub = real[real["delSize"] == ds].dropna(subset=["instability_g"])
        if sub.empty:
            continue
        colour = DELSIZE_PALETTE.get(int(ds), "#444444")
        marker = DELSIZE_MARKERS.get(int(ds), "o")
        # The two darkest traces benefit from a faintly heavier line so the
        # 0 % / 10 % light-pink end stays readable too.
        line_w = 1.6 if int(ds) >= 25 else 1.3
        ax.fill_between(
            sub["k"], sub["instability_low_g"], sub["instability_high_g"],
            facecolor=colour, alpha=0.25, linewidth=0,
        )
        ax.plot(
            sub["k"], sub["instability_g"],
            linestyle="-", marker=marker, markersize=4,
            color=colour, markeredgecolor="black", markeredgewidth=0.4,
            linewidth=line_w, label=f"delSize {ds} %",
        )

    # ── Null d-ceiling (deep navy + square markers) ──────────────────
    null_finite = null.dropna(subset=["instability_null_g"]).sort_values("k")
    ax.fill_between(
        null_finite["k"],
        null_finite["instability_null_low_g"],
        null_finite["instability_null_high_g"],
        facecolor=NULL_COLOUR, alpha=0.18, linewidth=0,
    )
    ax.plot(
        null_finite["k"], null_finite["instability_null_g"],
        linestyle="--", marker=NULL_MARKER, markersize=4,
        color=NULL_COLOUR, markeredgecolor="black", markeredgewidth=0.4,
        linewidth=1.4, label="Uniform null (R3 d-ceiling)",
    )

    # ── Axes ──────────────────────────────────────────────────────────
    ax.set_yscale("asinh", linear_width=args.asinh_linear_width)
    ax.set_xlabel("k-number")
    ax.set_ylabel(r"Centroid drift, Hungarian-matched (g)")
    ax.axvline(args.chosen_k, color=WONG["vermilion"], linestyle=":",
               linewidth=1.0, alpha=0.7)
    # x-axis tick set lifted from the existing internal-metrics
    # figure's Panel A so all panels of the composite share the same
    # k-grid visually.
    ax.set_xticks([2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xlim(1.5, 51.5)
    ax.tick_params(axis="x", labelsize="x-small")
    ax.legend(fontsize="x-small", loc="best", frameon=False)
    ax.grid(axis="y", which="both", linestyle=":", linewidth=0.4,
            color="#888888", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()

    combined = real.merge(
        null[["k", "instability_null_g",
              "instability_null_low_g", "instability_null_high_g"]],
        on="k", how="outer",
    )
    save_figure(fig, args.stem, args.output_dir, data=combined)
    print(
        f"\nWrote {args.output_dir / args.stem}.{{svg,png,csv}}  "
        f"(figsize = {args.figsize[0]} × {args.figsize[1]} in, "
        f"yscale = asinh)",
    )


if __name__ == "__main__":
    main()
