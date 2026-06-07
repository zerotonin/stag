#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.replot_internal_metrics_4reductions              ║
# ║  « Figure 2 overlay across all four leave-out reductions »       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Aggregates CH, Stability, Inertia for every fit across the      ║
# ║  four contiguous-leave-out reductions (delSize 0, 10, 25, 50)    ║
# ║  and renders a single Figure 2 with one trace + IQR band per     ║
# ║  reduction.  Silhouette stays as the rigorous delSize_0-only     ║
# ║  trace from the cached figure2_internal_metrics.csv (Panel C).   ║
# ║                                                                  ║
# ║  Inertia is computed centroid-only via nearest-centroid          ║
# ║  assignment on a fixed random subsample of the MaxAbs feature    ║
# ║  matrix — no per-fit labels.npy required (the original Aoraki    ║
# ║  sweep only mirrored ~24 labels per delSize/k, so a full         ║
# ║  per-fit IQR is not derivable from labels alone).  Subsampled    ║
# ║  W(k) is scaled by (n_full / n_subsample) to match the           ║
# ║  manuscript's axis.                                              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Overlay Figure 2 across all four contiguous-leave-out reductions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from stag.constants import (
    CLUSTER_RESULTS_DIR,
    MAXABS_CLUSTERING_INPUT,
    RESULTS_DIR_DEFAULT,
    WONG,
    apply_figure_defaults,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta-dir", type=Path, default=CLUSTER_RESULTS_DIR,
        help="Directory of clustering-meta JSON files.",
    )
    parser.add_argument(
        "--data-file", type=Path, default=MAXABS_CLUSTERING_INPUT,
        help="MaxAbs-scaled feature matrix matching the saved centroids.",
    )
    parser.add_argument(
        "--silhouette-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "tables" / "figure2_internal_metrics.csv",
        help="Cached delSize_0 silhouette CSV (Panel C source).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR_DEFAULT,
        help="Top-level results directory.",
    )
    parser.add_argument(
        "--subsample", type=int, default=2_000_000,
        help="Subsample size for centroid-based inertia (default 2M).",
    )
    parser.add_argument("--chosen-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
#  Inertia via nearest-centroid assignment on a subsample
# ─────────────────────────────────────────────────────────────────

def _sse_on_subsample(
    centroids: np.ndarray, X_sub: np.ndarray, X_sub_norm_sum: float,
) -> float:
    """Sum-of-squared-distance to the nearest centroid (decomposed form).

    ||x - c||² = ||x||² - 2 x·c + ||c||².  ||x||² sums to a fit-
    independent constant ``X_sub_norm_sum``; per-fit we only need
    ``2 x·c`` and ``||c||²``.  This avoids the O(N·k·d) broadcast.
    """
    c_norm = (centroids * centroids).sum(axis=1)         # (k,)
    score = X_sub @ centroids.T                          # (N, k)
    # ||x - c||² for nearest c: argmin over (-2 x·c + ||c||²) → min over (-2 score + c_norm).
    per_sample = (-2.0 * score + c_norm).min(axis=1)     # (N,)
    return float(X_sub_norm_sum + per_sample.sum())


def collect_fits(meta_dir: Path) -> pd.DataFrame:
    """Walk meta JSONs and return tall (delSize, k, cut_pos, ch, centroids_path) DataFrame."""
    rows = []
    for delsize_dir in sorted(meta_dir.glob("delSize_*")):
        for k_dir in sorted(delsize_dir.glob("k_*")):
            for meta_path in sorted(k_dir.glob("*_meta_*.json")):
                d = json.loads(meta_path.read_text())
                centroids_path = meta_path.parent / "centroids" / (
                    meta_path.stem.replace("_meta_", "_centroids_") + ".npy"
                )
                rows.append({
                    "delSize": int(d["reduction_percent"]),
                    "k": len(d["centroids"]),
                    "cut_pos": float(d.get("cut_position_percent", 0.0)),
                    "ch": float(d["calinski_harabasz_score"]),
                    "meta_path": str(meta_path),
                    "centroids_path": str(centroids_path),
                    "centroids_inline": d["centroids"],
                })
    return pd.DataFrame(rows)


def compute_inertias(df: pd.DataFrame, X_sub: np.ndarray, n_full: int) -> pd.Series:
    """For each row, compute SSE on the subsample, scale to full-data equivalent."""
    scale = n_full / X_sub.shape[0]
    X_sub_norm_sum = float((X_sub * X_sub).sum())  # constant across fits
    inertias = np.empty(len(df), dtype=np.float64)
    for i, row in enumerate(tqdm(df.itertuples(), total=len(df),
                                  desc="centroid-only inertia")):
        c = np.asarray(row.centroids_inline, dtype=np.float32)
        inertias[i] = _sse_on_subsample(c, X_sub, X_sub_norm_sum) * scale
    return pd.Series(inertias, index=df.index, name="inertia")


# ─────────────────────────────────────────────────────────────────
#  Hungarian-matched centroid drift (instability) per (k, delSize)
# ─────────────────────────────────────────────────────────────────

def compute_instability(df: pd.DataFrame) -> pd.Series:
    """Per-fit Hungarian-matched centroid drift vs the basin median.

    Within each (delSize, k) group, treat one fit as the reference
    (highest CH) and compute, for every other fit, the Hungarian-
    matched mean centroid displacement.  Returns one value per fit
    (NaN for the reference).
    """
    inst = np.full(len(df), np.nan)
    for (ds, k), grp in df.groupby(["delSize", "k"]):
        if len(grp) < 2:
            continue
        # Take the highest-CH fit as the basin reference.
        ref_idx = grp["ch"].idxmax()
        ref_c = np.asarray(grp.loc[ref_idx, "centroids_inline"], dtype=np.float32)
        for i, row in grp.iterrows():
            if i == ref_idx:
                continue
            other_c = np.asarray(row["centroids_inline"], dtype=np.float32)
            # Hungarian on pairwise centroid distances.
            D = np.linalg.norm(
                ref_c[:, None, :] - other_c[None, :, :], axis=2,
            )
            r, c = linear_sum_assignment(D)
            inst[df.index.get_loc(i)] = float(D[r, c].mean())
    return pd.Series(inst, index=df.index, name="instability")


# ─────────────────────────────────────────────────────────────────
#  Aggregation + plotting
# ─────────────────────────────────────────────────────────────────

def aggregate_per_k(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.groupby(["delSize", "k"])[value_col]
          .agg(median="median",
               low=lambda s: float(np.nanquantile(s, 0.25)),
               high=lambda s: float(np.nanquantile(s, 0.75)))
          .reset_index()
          .rename(columns={"median": value_col,
                           "low": f"{value_col}_low",
                           "high": f"{value_col}_high"})
    )


def render_figure(
    ch_agg: pd.DataFrame, inst_agg: pd.DataFrame, inertia_agg: pd.DataFrame,
    silhouette_df: pd.DataFrame, chosen_k: int, output_dir: Path,
) -> None:
    apply_figure_defaults()
    delsize_palette = {
        0:  WONG["blue"],
        10: WONG["bluish_green"],
        25: WONG["orange"],
        50: WONG["reddish_purple"],
    }

    fig, ((ax_ch, ax_inst), (ax_sil, ax_W)) = plt.subplots(
        2, 2, figsize=(9, 7.2),
    )

    def _draw_overlay(ax, agg, value_col, ylabel, title, log_y=False):
        for ds, sub in agg.groupby("delSize"):
            sub = sub.sort_values("k")
            colour = delsize_palette[ds]
            ax.fill_between(
                sub["k"], sub[f"{value_col}_low"], sub[f"{value_col}_high"],
                facecolor=colour, alpha=0.20, linewidth=0,
            )
            ax.plot(
                sub["k"], sub[value_col], "-o", color=colour, markersize=3,
                linewidth=1.3, label=f"delSize {ds} %",
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.axvline(chosen_k, color=WONG["vermilion"], linestyle="--",
                   linewidth=1.0, alpha=0.7)

    _draw_overlay(ax_ch, ch_agg, "ch",
                  "CH index (higher is better)", "(A) Quality")
    _draw_overlay(ax_inst, inst_agg, "instability",
                  "Hungarian-matched drift\n(lower is better)", "(B) Stability",
                  log_y=True)

    # Panel C — silhouette stays at delSize_0 only.
    sil_finite = silhouette_df.dropna(subset=["silhouette"]).sort_values("k")
    ax_sil.fill_between(
        sil_finite["k"], sil_finite["silhouette_low"], sil_finite["silhouette_high"],
        facecolor=WONG["blue"], alpha=0.35, linewidth=0,
    )
    ax_sil.plot(sil_finite["k"], sil_finite["silhouette"], "-o",
                color=WONG["blue"], markersize=3, linewidth=1.3,
                label="delSize 0 % (rigorous)")
    ax_sil.set_title("(C) Silhouette")
    ax_sil.set_ylabel(r"Mean silhouette ($\bar{s}$)")
    ax_sil.set_xlabel("k")
    ax_sil.axvline(chosen_k, color=WONG["vermilion"], linestyle="--",
                   linewidth=1.0, alpha=0.7)

    _draw_overlay(ax_W, inertia_agg, "inertia",
                  r"$W(k)$", "(D) Inertia / Elbow")
    ax_W.set_xlabel("k")

    # Kneedle elbow on the delSize_0 trace only.
    ds0 = inertia_agg[inertia_agg["delSize"] == 0].sort_values("k")
    if len(ds0) >= 3:
        finite = ds0.dropna(subset=["inertia"])
        if len(finite) >= 3:
            knee = KneeLocator(finite["k"], finite["inertia"],
                                curve="convex", direction="decreasing", S=1.0)
            if knee.knee is not None:
                k_e = int(knee.knee)
                y_e = float(finite.loc[finite["k"] == k_e, "inertia"].iloc[0])
                ax_W.scatter([k_e], [y_e], s=70, color=WONG["orange"],
                             zorder=5, label=f"Kneedle elbow (delSize 0, k = {k_e})")

    for ax in (ax_ch, ax_inst, ax_sil, ax_W):
        ax.set_xticks([2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50])
        ax.tick_params(axis="x", labelsize="x-small")
    ax_ch.legend(fontsize="x-small", loc="best", frameon=False)
    ax_W.legend(fontsize="x-small", loc="best", frameon=False)

    fig.tight_layout()

    # Build combined CSV.
    combined = (
        ch_agg.merge(inst_agg, on=["delSize", "k"], how="outer")
              .merge(inertia_agg, on=["delSize", "k"], how="outer")
    )
    save_figure(
        fig, "figure2_internal_metrics_4reductions",
        output_dir / "figures", data=combined,
    )
    print(f"\nWrote {output_dir/'figures'/'figure2_internal_metrics_4reductions'}.{{svg,png,csv}}")


# ─────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    print(f"Walking meta JSONs under {args.meta_dir} ...")
    df = collect_fits(args.meta_dir)
    print(f"  collected {len(df)} fits across delSizes {sorted(df['delSize'].unique())}")
    print(f"  k range: {df['k'].min()}..{df['k'].max()}")
    print(f"  per-(delSize, k) fit-count median: "
          f"{df.groupby(['delSize','k']).size().median():.0f}")

    print(f"\nLoading data + sampling {args.subsample:,} rows ...")
    data = np.load(args.data_file, mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(data.shape[0], args.subsample, replace=False)
    idx.sort()
    X_sub = data[idx].astype(np.float32, copy=True)
    print(f"  subsample shape: {X_sub.shape}, full size: {data.shape[0]:,}")

    print(f"\nComputing centroid-only inertia for {len(df)} fits ...")
    df["inertia"] = compute_inertias(df, X_sub, n_full=data.shape[0])

    print("\nComputing Hungarian-matched instability ...")
    df["instability"] = compute_instability(df)

    print("\nAggregating per (delSize, k) ...")
    ch_agg = aggregate_per_k(df, "ch")
    inst_agg = aggregate_per_k(df, "instability")
    inertia_agg = aggregate_per_k(df, "inertia")

    print(f"\nLoading delSize_0 silhouette from {args.silhouette_csv} ...")
    sil_df = pd.read_csv(args.silhouette_csv)
    sil_df = sil_df[["k", "silhouette", "silhouette_low", "silhouette_high"]]
    print(f"  silhouette rows: {len(sil_df)}  ({sil_df['silhouette'].notna().sum()} non-NaN)")

    print("\nRendering Figure 2 (4-reduction overlay) ...")
    render_figure(ch_agg, inst_agg, inertia_agg, sil_df,
                  chosen_k=args.chosen_k, output_dir=args.output_dir)

    df.drop(columns=["centroids_inline"], inplace=True)
    df.to_csv(args.output_dir / "tables" / "figure2_internal_metrics_4reductions_per_fit.csv",
              index=False)
    print(f"Wrote per-fit table: "
          f"{args.output_dir / 'tables' / 'figure2_internal_metrics_4reductions_per_fit.csv'}")


if __name__ == "__main__":
    main()
