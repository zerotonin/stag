#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.run_internal_metrics                             ║
# ║  « driver for the revised Figure 2 + supplementary table »       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Walks the meta-JSON tree produced by the SLURM clustering       ║
# ║  sweep, recomputes inertia where it is missing, computes         ║
# ║  stratified silhouette across the k grid, locates the Kneedle    ║
# ║  elbow, and emits:                                               ║
# ║                                                                  ║
# ║    results/tables/figure2_internal_metrics.csv  (supp. table)    ║
# ║    results/figures/figure2_internal_metrics.svg  +  .png  +  .csv║
# ║                                                                  ║
# ║  Designed to run on the workstation against the saved Aoraki     ║
# ║  artefacts.  No GPU required.                                    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Driver: compute internal metrics across the k grid and render Figure 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from stag.clustering.internal_metrics import (
    compute_silhouette_stratified,
    locate_elbow_kneedle,
    recompute_inertia_for_meta_dir,
    selection_summary,
)
from stag.clustering.meta_analysis import ClusterMetaAnalysis
from stag.clustering.plotting import plot_internal_metrics_panel
from stag.constants import (
    CLUSTER_RESULTS_DIR,
    MAXABS_CLUSTERING_INPUT,
    RESULTS_DIR_DEFAULT,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute internal metrics across the k grid and render Figure 2.",
    )
    parser.add_argument("--meta-dir", type=Path, default=CLUSTER_RESULTS_DIR,
                        help="Directory of clustering-meta JSON files "
                             "(default: stag.constants.CLUSTER_RESULTS_DIR).")
    parser.add_argument("--data-file", type=Path, default=MAXABS_CLUSTERING_INPUT,
                        help="MaxAbs-scaled feature matrix matching the saved "
                             "centroids (default: stag.constants.MAXABS_CLUSTERING_INPUT). "
                             "Must be in the same per-column scale as the centroids "
                             "or inertia will be on a different scale to the fit.")
    parser.add_argument("--reduction-percent", type=float, default=0.0,
                        help="Slice of fits to use (default 0 = full-data runs).")
    parser.add_argument("--silhouette-per-cluster", type=int, default=5000)
    parser.add_argument("--silhouette-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chosen-k", type=int, default=8,
                        help="k highlighted across all panels (default 8).")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR_DEFAULT,
                        help="Top-level results directory.")
    parser.add_argument("--overwrite-meta", action="store_true",
                        help="Write the recomputed inertia back into each JSON.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Process-pool size for the inertia back-fill "
                             "(default 1 = sequential).  8–12 is a good "
                             "starting point on the 20-CPU workstation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # ─── Inertia back-fill ────────────────────────────────────────────
    #
    # Restrict to the reduction_percent the silhouette pass actually
    # uses.  Saves ~75 % of the work and keeps memory bounded on the
    # 62 GB workstation — non-zero reduction_percent fits still call
    # np.delete and materialise a multi-GB copy per worker.
    print(f"Recomputing inertia for reduction_percent={args.reduction_percent} "
          f"fits (workers = {args.workers}) ...")
    inertia_df = recompute_inertia_for_meta_dir(
        args.meta_dir, args.data_file,
        overwrite=args.overwrite_meta,
        workers=args.workers,
        reduction_percents=[args.reduction_percent],
    )

    # ─── Load meta DataFrame (CH + instability) ──────────────────────
    print("Loading meta-analysis DataFrame ...")
    meta = ClusterMetaAnalysis(args.meta_dir)
    meta.analyze()
    full = meta.df.copy()
    full = full[full["reduction_percent"] == args.reduction_percent]

    if "inertia" in full.columns:
        # With --overwrite-meta the workers wrote inertia back into each
        # JSON and ClusterMetaAnalysis.load_data() picked it up on
        # re-read.  No merge needed; values are identical to inertia_df
        # by construction.
        pass
    elif inertia_df.empty:
        print("WARNING: no inertia values back-filled; falling back to NaN inertia.")
        full["inertia"] = float("nan")
    else:
        # Join inertia back onto the meta DataFrame by file_path.
        full = full.merge(
            inertia_df[["file_path", "inertia"]],
            on="file_path", how="left",
        )

    # ─── Per-k aggregation (median + IQR across cut positions) ──────
    #
    # Each metric: median across the 50 cut positions at delSize_0,
    # with the inter-quartile range (25th / 75th percentile) as the
    # dispersion band.  Matches the manuscript's existing Figure 2A
    # convention ("median across runs shown as a line; shaded band
    # shows IQR") — re-reviewers see consistency between the original
    # and revised figure.  The dispersion at delSize_0 is small enough
    # that 95 % CI bands were invisible at figure scale; IQR communicates
    # the same "tightly converged" story honestly.
    def _q025(x): return float(np.nanquantile(x, 0.25))
    def _q975(x): return float(np.nanquantile(x, 0.75))

    grouped = full.groupby("k_number")
    per_k = grouped.agg(
        ch=("calinski_harabasz_score", "median"),
        ch_low=("calinski_harabasz_score", _q025),
        ch_high=("calinski_harabasz_score", _q975),
        instability=("instability", "median"),
        instability_low=("instability", _q025),
        instability_high=("instability", _q975),
        inertia=("inertia", "median"),
        inertia_low=("inertia", _q025),
        inertia_high=("inertia", _q975),
    ).reset_index().rename(columns={"k_number": "k"}).sort_values("k")

    # ─── Stratified Silhouette at every k (load labels) ──────────────
    #
    # CH is often near-identical across the 50 cut positions of one k
    # (the SLURM sweep converges to the same global optimum from
    # different initialisations).  Picking strictly the idxmax run is
    # fragile because the copy script's iteration order does not
    # match pandas' tie-break, so we walk the CH-sorted candidates
    # and use the first one whose labels file is on disk locally.
    X = np.load(args.data_file)
    silhouettes: list[float] = []
    silhouettes_low: list[float] = []
    silhouettes_high: list[float] = []
    for k in tqdm(per_k["k"], desc="silhouette per k"):
        cands = full[full["k_number"] == k]
        nan_row = (np.nan, np.nan, np.nan)

        def _append(triple):
            silhouettes.append(triple[0])
            silhouettes_low.append(triple[1])
            silhouettes_high.append(triple[2])

        if cands.empty:
            _append(nan_row)
            continue
        ordered = cands.sort_values(
            "calinski_harabasz_score", ascending=False,
        )
        labels_path: Path | None = None
        for _, row in ordered.iterrows():
            candidate = (
                Path(row["file_path"]).parent
                / "labels"
                / Path(row["file_path"]).name
                    .replace("_meta_", "_labels_").replace(".json", ".npy")
            )
            if candidate.exists():
                labels_path = candidate
                break
        if labels_path is None:
            _append(nan_row)
            continue
        labels = np.load(labels_path)
        if labels.shape[0] != X.shape[0]:
            _append(nan_row)
            continue
        result = compute_silhouette_stratified(
            X, labels,
            n_per_cluster=args.silhouette_per_cluster,
            n_repeats=args.silhouette_repeats,
            rng=rng,
        )
        # IQR over the per-repeat distribution (n_repeats samples of
        # the mean silhouette).  Same 25/75 percentile choice as the
        # other panels for consistency with manuscript Figure 2A.
        per_repeat = result["per_repeat"]
        _append((
            result["mean_silhouette"],
            float(np.quantile(per_repeat, 0.25)),
            float(np.quantile(per_repeat, 0.75)),
        ))

    summary = selection_summary(
        per_k["k"].tolist(),
        ch=per_k["ch"].tolist(),
        ch_low=per_k["ch_low"].tolist(),
        ch_high=per_k["ch_high"].tolist(),
        instability=per_k["instability"].tolist(),
        instability_low=per_k["instability_low"].tolist(),
        instability_high=per_k["instability_high"].tolist(),
        silhouette=silhouettes,
        silhouette_low=silhouettes_low,
        silhouette_high=silhouettes_high,
        inertia=per_k["inertia"].tolist(),
        inertia_low=per_k["inertia_low"].tolist(),
        inertia_high=per_k["inertia_high"].tolist(),
    )

    # ─── Kneedle elbow ───────────────────────────────────────────────
    finite_W = summary.dropna(subset=["inertia"])
    elbow = locate_elbow_kneedle(
        finite_W["k"].tolist(), finite_W["inertia"].tolist(),
    )
    elbow_k = elbow["elbow_k"]

    # ─── Outputs ─────────────────────────────────────────────────────
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(tables_dir / "figure2_internal_metrics.csv", index=False)

    fig = plot_internal_metrics_panel(
        summary, elbow_k=elbow_k, chosen_k=args.chosen_k,
    )
    save_figure(fig, "figure2_internal_metrics", figures_dir, data=summary)

    print(f"  Kneedle elbow at k = {elbow_k}")
    print(f"  Manuscript chosen k = {args.chosen_k}")
    print(f"  Table:  {tables_dir / 'figure2_internal_metrics.csv'}")
    print(f"  Figure: {figures_dir / 'figure2_internal_metrics.svg'}")


if __name__ == "__main__":
    main()
