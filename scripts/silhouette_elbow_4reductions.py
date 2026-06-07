#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.silhouette_elbow_4reductions                     ║
# ║  « silhouette + Kneedle elbow per delSize, custom palette »      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Computes stratified silhouette at every (delSize, k) by         ║
# ║  deriving labels via nearest-centroid assignment on the per-     ║
# ║  fit data slice (no per-fit labels.npy required).  Reuses the    ║
# ║  inertia values from figure2_internal_metrics_4reductions.csv    ║
# ║  for the elbow panel, locating a Kneedle elbow per delSize.      ║
# ║                                                                  ║
# ║  Output is a single 1×2 figure: silhouette overlay on the left,  ║
# ║  inertia overlay with 4 elbow dots on the right, drawn in the    ║
# ║  user-supplied palette (light pink → dark purple).               ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Silhouette + Kneedle elbow at every delSize, with overlay colours."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# kneed (KneeLocator) is imported lazily inside the elbow-detection
# branch of main() so that SLURM array tasks running in --csv-only
# mode do not need to install it on the cluster.
from stag.clustering.internal_metrics import compute_silhouette_stratified
from stag.clustering.kmeans import shrink_data
from stag.constants import (
    CLUSTER_RESULTS_DIR,
    MAXABS_CLUSTERING_INPUT,
    RESULTS_DIR_DEFAULT,
    apply_figure_defaults,
    save_figure,
)

PALETTE: dict[int, str] = {
    0:  "#EECCC9",
    10: "#DB97AA",
    25: "#A75491",
    50: "#291E38",
}

# Per-leave-out-reduction marker convention (lab canonical):
#  0 %  -> circle, 10 % -> upward triangle,
#  25 % -> downward triangle, 50 % -> left-ward triangle.
MARKERS: dict[int, str] = {
    0:  "o",
    10: "^",
    25: "v",
    50: "<",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-dir", type=Path, default=CLUSTER_RESULTS_DIR)
    parser.add_argument("--data-file", type=Path, default=MAXABS_CLUSTERING_INPUT)
    parser.add_argument(
        "--inertia-csv", type=Path,
        default=RESULTS_DIR_DEFAULT / "figures"
                / "figure2_internal_metrics_4reductions.csv",
        help="Combined per-(delSize,k) summary from the 4-reduction overlay run.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--silhouette-per-cluster", type=int, default=5000)
    parser.add_argument("--silhouette-repeats", type=int, default=20)
    parser.add_argument("--max-k", type=int, default=None,
                        help="Skip representatives with k > max-k.  Silhouette is "
                             "O(n_per_cluster² · k²) and the high-k tail dominates "
                             "runtime without adding rebuttal information.")
    parser.add_argument("--only-delsize", type=int, default=None,
                        help="Restrict to a single delSize value (use with "
                             "--only-k for SLURM array task mode).")
    parser.add_argument("--only-k", type=int, default=None,
                        help="Restrict to a single k value (use with "
                             "--only-delsize for SLURM array task mode).")
    parser.add_argument("--csv-only", action="store_true",
                        help="Skip figure rendering, emit silhouette CSV only.  "
                             "Used in SLURM array tasks where the final figure "
                             "is rendered by a separate merge step.")
    parser.add_argument("--silhouette-csv-out", type=Path, default=None,
                        help="Path for the silhouette CSV (default: "
                             "tables/silhouette_per_delSize_k.csv).  Useful when "
                             "running array tasks so each task emits its own file.")
    parser.add_argument("--chosen-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def pick_representative_per_kdelsize(meta_dir: Path) -> pd.DataFrame:
    """For every (delSize, k), pick the meta JSON with the highest CH.

    The representative fit is the basin's modal converged partition
    — what we use to compute silhouette and the inertia curve.
    """
    rows = []
    for delsize_dir in sorted(meta_dir.glob("delSize_*")):
        for k_dir in sorted(delsize_dir.glob("k_*")):
            best: dict | None = None
            for meta_path in k_dir.glob("*_meta_*.json"):
                d = json.loads(meta_path.read_text())
                if best is None or d["calinski_harabasz_score"] > best["calinski_harabasz_score"]:
                    d["_meta_path"] = meta_path
                    best = d
            if best is None:
                continue
            rows.append({
                "delSize": int(best["reduction_percent"]),
                "k": len(best["centroids"]),
                "cut_pos": float(best.get("cut_position_percent", 0.0)),
                "ch": float(best["calinski_harabasz_score"]),
                "centroids_inline": best["centroids"],
            })
    return pd.DataFrame(rows).sort_values(["delSize", "k"]).reset_index(drop=True)


def assign_nearest(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Argmin nearest-centroid assignment via the decomposed distance form."""
    c_norm = (centroids * centroids).sum(axis=1)
    score = X @ centroids.T               # (N, k)
    return np.argmin(-2.0 * score + c_norm, axis=1).astype(np.int32)


def main() -> None:
    args = parse_args()
    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Picking representative fits per (delSize, k) under {args.meta_dir} ...")
    reps = pick_representative_per_kdelsize(args.meta_dir)
    if args.max_k is not None:
        before = len(reps)
        reps = reps[reps["k"] <= args.max_k].reset_index(drop=True)
        print(f"  capping at k <= {args.max_k}: kept {len(reps)} of {before} reps")
    if args.only_delsize is not None:
        before = len(reps)
        reps = reps[reps["delSize"] == args.only_delsize].reset_index(drop=True)
        print(f"  restricting to delSize={args.only_delsize}: kept {len(reps)} of {before} reps")
    if args.only_k is not None:
        before = len(reps)
        reps = reps[reps["k"] == args.only_k].reset_index(drop=True)
        print(f"  restricting to k={args.only_k}: kept {len(reps)} of {before} reps")
    if reps.empty:
        raise SystemExit("No representative fits selected after filters; nothing to do.")
    print(f"  selected {len(reps)} representative fits "
          f"({reps['delSize'].nunique()} delSizes × "
          f"{reps['k'].nunique()} k values)")

    print(f"\nLoading {args.data_file.name} (mmap) ...")
    data = np.load(args.data_file, mmap_mode="r")
    print(f"  shape: {data.shape}")

    rng = np.random.default_rng(args.seed)
    sil_rows = []
    for row in tqdm(reps.itertuples(), total=len(reps),
                    desc="silhouette per (delSize, k)"):
        # 1) slice the data to this fit's leave-out window.
        X_used = shrink_data(np.asarray(data), row.delSize, row.cut_pos)
        X_used = X_used.astype(np.float32, copy=False)
        # 2) derive labels via nearest-centroid (the basin's converged labels).
        centroids = np.asarray(row.centroids_inline, dtype=np.float32)
        labels = assign_nearest(X_used, centroids)
        # 3) stratified silhouette on (X_used, labels).
        result = compute_silhouette_stratified(
            X_used, labels,
            n_per_cluster=args.silhouette_per_cluster,
            n_repeats=args.silhouette_repeats,
            rng=rng,
        )
        per_repeat = result["per_repeat"]
        sil_rows.append({
            "delSize": row.delSize,
            "k": row.k,
            "silhouette":      float(result["mean_silhouette"]),
            "silhouette_low":  float(np.quantile(per_repeat, 0.25)),
            "silhouette_high": float(np.quantile(per_repeat, 0.75)),
        })

    sil_df = pd.DataFrame(sil_rows)
    sil_csv = args.silhouette_csv_out or (tables_dir / "silhouette_per_delSize_k.csv")
    sil_csv.parent.mkdir(parents=True, exist_ok=True)
    sil_df.to_csv(sil_csv, index=False)
    print(f"\nWrote silhouette table: {sil_csv}")

    if args.csv_only:
        print("--csv-only set; skipping figure rendering.")
        return

    # ─── Inertia / elbow per delSize ─────────────────────────────────
    # KneeLocator is only needed when the figure is rendered (i.e. not
    # in --csv-only mode), so the import is deferred to here.
    from kneed import KneeLocator

    inertia_summary = pd.read_csv(args.inertia_csv)
    # That file has columns: delSize, k, ch, ch_low, ch_high, instability, ...,
    # inertia, inertia_low, inertia_high.
    elbows: dict[int, int] = {}
    for ds, sub in inertia_summary.groupby("delSize"):
        sub = sub.sort_values("k").dropna(subset=["inertia"])
        if len(sub) >= 3:
            knee = KneeLocator(
                sub["k"], sub["inertia"],
                curve="convex", direction="decreasing", S=1.0,
            )
            if knee.knee is not None:
                elbows[int(ds)] = int(knee.knee)
    print(f"\nKneedle elbows per delSize: {elbows}")

    # ─── Figure — 2x1 vertical with shared x-axis ────────────────────
    apply_figure_defaults()
    fig, (ax_sil, ax_W) = plt.subplots(
        2, 1, figsize=(7.5, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    # Silhouette overlay.
    for ds, colour in PALETTE.items():
        sub = sil_df[sil_df["delSize"] == ds].sort_values("k")
        if sub.empty:
            continue
        ax_sil.fill_between(
            sub["k"], sub["silhouette_low"], sub["silhouette_high"],
            facecolor=colour, alpha=0.30, linewidth=0,
        )
        ax_sil.plot(
            sub["k"], sub["silhouette"],
            linestyle="-", marker=MARKERS[ds],
            color=colour, markersize=5, markeredgecolor="black",
            markeredgewidth=0.4, linewidth=1.4,
            label=f"delSize {ds} %",
        )
    ax_sil.set_title("(C) Silhouette per leave-out reduction")
    ax_sil.set_ylabel(r"Mean silhouette ($\bar{s}$)")
    ax_sil.axvline(args.chosen_k, color="black", linestyle="--",
                   linewidth=0.8, alpha=0.5)
    ax_sil.legend(fontsize="small", loc="best", frameon=False)

    # Inertia overlay + elbows.
    for ds, colour in PALETTE.items():
        sub = inertia_summary[inertia_summary["delSize"] == ds].sort_values("k")
        if sub.empty:
            continue
        ax_W.fill_between(
            sub["k"], sub["inertia_low"], sub["inertia_high"],
            facecolor=colour, alpha=0.30, linewidth=0,
        )
        ax_W.plot(
            sub["k"], sub["inertia"],
            linestyle="-", marker=MARKERS[ds],
            color=colour, markersize=5, markeredgecolor="black",
            markeredgewidth=0.4, linewidth=1.4,
            label=f"delSize {ds} %",
        )
        if ds in elbows:
            k_e = elbows[ds]
            row_e = sub[sub["k"] == k_e]
            if not row_e.empty:
                y_e = float(row_e["inertia"].iloc[0])
                ax_W.scatter(
                    [k_e], [y_e],
                    s=100, facecolor="white", edgecolor=colour, linewidth=2.0,
                    zorder=6, label=f"Kneedle k = {k_e} (delSize {ds} %)",
                )
    ax_W.set_title("(D) Inertia / Kneedle elbow per leave-out reduction")
    ax_W.set_xlabel("k")
    ax_W.set_ylabel(r"$W(k)$")
    ax_W.axvline(args.chosen_k, color="black", linestyle="--",
                 linewidth=0.8, alpha=0.5)
    ax_W.legend(fontsize="x-small", loc="best", frameon=False)

    # Shared x-axis: ticks only need to be set on the bottom panel.
    ax_W.set_xticks([2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50])
    ax_W.tick_params(axis="x", labelsize="x-small")

    fig.tight_layout()

    combined = sil_df.merge(
        inertia_summary[["delSize", "k", "inertia", "inertia_low", "inertia_high"]],
        on=["delSize", "k"], how="outer",
    )
    save_figure(
        fig, "silhouette_elbow_per_delSize", figures_dir, data=combined,
    )
    print(f"\nWrote {figures_dir/'silhouette_elbow_per_delSize'}.{{svg,png,csv}}")


if __name__ == "__main__":
    main()
