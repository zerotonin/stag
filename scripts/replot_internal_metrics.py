#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.replot_internal_metrics                          ║
# ║  « re-render Figure 2 from the cached summary CSV »              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  run_internal_metrics.py writes the per-k summary to             ║
# ║  results/tables/figure2_internal_metrics.csv on every full run.  ║
# ║  Re-rendering the figure from that cache takes seconds, vs       ║
# ║  ~20 min to recompute silhouette / inertia from scratch.         ║
# ║                                                                  ║
# ║  Use this for plot-style iteration (alpha, palette, axis         ║
# ║  scaling, elbow annotation) without paying the full pipeline     ║
# ║  cost.  The script reads the same CSV the full pipeline writes,  ║
# ║  re-locates the Kneedle elbow from the cached inertia column,    ║
# ║  and emits the same SVG + PNG + CSV trio next to the original.   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Re-render Figure 2 from the cached internal-metrics summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stag.clustering.internal_metrics import locate_elbow_kneedle
from stag.clustering.plotting import plot_internal_metrics_panel
from stag.constants import RESULTS_DIR_DEFAULT, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "tables" / "figure2_internal_metrics.csv",
        help="Cached per-k summary CSV from run_internal_metrics.py "
             "(default: results/tables/figure2_internal_metrics.csv).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR_DEFAULT,
        help="Top-level results directory (figures land in <output-dir>/figures).",
    )
    parser.add_argument(
        "--chosen-k", type=int, default=8,
        help="k highlighted across all panels (default 8).",
    )
    parser.add_argument(
        "--stem", type=str, default="figure2_internal_metrics",
        help="Output filename stem (default reuses figure2_internal_metrics).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.summary_csv.is_file():
        raise SystemExit(
            f"Summary CSV not found: {args.summary_csv}\n"
            f"Run scripts/run_internal_metrics.py at least once first.",
        )

    summary = pd.read_csv(args.summary_csv)
    print(f"Loaded summary: {args.summary_csv}  rows={len(summary)}")

    finite_W = summary.dropna(subset=["inertia"])
    elbow = locate_elbow_kneedle(
        finite_W["k"].tolist(), finite_W["inertia"].tolist(),
    )
    elbow_k = elbow["elbow_k"]
    print(f"  Kneedle elbow at k = {elbow_k}")
    print(f"  Manuscript chosen k = {args.chosen_k}")

    fig = plot_internal_metrics_panel(
        summary, elbow_k=elbow_k, chosen_k=args.chosen_k,
    )

    figures_dir = args.output_dir / "figures"
    save_figure(fig, args.stem, figures_dir, data=summary)
    print(f"  Figure: {figures_dir / f'{args.stem}.svg'}")


if __name__ == "__main__":
    main()
