#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.run_stability_null_block                         ║
# ║  « one (surrogate_seed, k) GPU block for the R3 stability null » ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Loads one pre-built uniform MaxAbs-box surrogate, runs ``N``    ║
# ║  RAPIDS cuML k-means fits with distinct random states, computes  ║
# ║  the Hungarian-matched centroid drift across the ensemble, and   ║
# ║  emits a single CSV row per fit:                                 ║
# ║                                                                  ║
# ║     surrogate_seed, k, kmeans_seed, ch, instability_null,        ║
# ║     fit_duration_s                                               ║
# ║                                                                  ║
# ║  Mirrors the ``delSize_0`` real-run protocol (no leave-out;      ║
# ║  ensemble drift driven by the k-means RNG) so the resulting      ║
# ║  band is directly comparable on Panel B of the figure.           ║
# ║                                                                  ║
# ║  Designed for SLURM array submission: 10 surrogate seeds × 14    ║
# ║  k-values = 140 tasks; see ``slurm/stability_null_array.sh``.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Run one (surrogate_seed, k) block of the R3 stability null."""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stag.analysis.stability_null import hungarian_centroid_drift


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surrogate", type=Path, required=True,
        help="Pre-built null_uniform_seed<SS>.npy for this block.",
    )
    parser.add_argument(
        "--surrogate-seed", type=int, required=True,
        help="Seed index used to build the surrogate (for logging + CSV).",
    )
    parser.add_argument(
        "--k", type=int, required=True,
        help="Number of k-means clusters to fit on the surrogate.",
    )
    parser.add_argument(
        "--n-fits", type=int, default=20,
        help="K-means fits per (seed, k) block (different random states).",
    )
    parser.add_argument(
        "--kmeans-seed-base", type=int, default=0,
        help="Base random_state; each fit uses base + fit_idx.",
    )
    parser.add_argument(
        "--ch-subsample", type=int, default=200_000,
        help="Sub-sample size for the Calinski-Harabasz score "
             "(full-N is wastefully expensive at N = 2e8).",
    )
    parser.add_argument(
        "--unscale-csv", type=Path, default=None,
        help="Path to the MaxAbs divisors CSV.  When provided, "
             "centroids are multiplied by the per-column divisors "
             "before the Hungarian distance calc, so the per-fit "
             "instability values land in physical (g-force) units "
             "rather than MaxAbs space.  Pass "
             "``stag.constants.MAXABS_SCALER_CSV`` to match the "
             "deer-2024 pipeline.",
    )
    parser.add_argument(
        "--output-csv", type=Path, required=True,
        help="Destination CSV for the per-fit rows.",
    )
    return parser.parse_args()


def _import_gpu_stack():
    """Late-import RAPIDS so this script imports on CPU-only nodes too."""
    try:
        import cupy as cp
        from cuml.cluster import KMeans
    except ImportError as exc:
        raise RuntimeError(
            "RAPIDS cuML is required for the stability null GPU pass. "
            "Activate the conda env at $STAG_HPC_CONDA_PY before running.",
        ) from exc
    return cp, KMeans


def main() -> None:
    args = parse_args()
    cp, KMeans = _import_gpu_stack()
    from sklearn.metrics import calinski_harabasz_score, silhouette_score

    print(f"[{datetime.datetime.now()}] Loading surrogate {args.surrogate}")
    surrogate_cpu = np.load(args.surrogate, mmap_mode="r")
    print(
        f"  surrogate shape = {surrogate_cpu.shape}, "
        f"dtype = {surrogate_cpu.dtype}",
    )

    print(f"[{datetime.datetime.now()}] Uploading surrogate to GPU")
    surrogate_gpu = cp.asarray(surrogate_cpu)

    centroids_list: list[np.ndarray] = []
    ch_scores: list[float] = []
    inertias: list[float] = []
    silhouettes: list[float] = []
    durations: list[float] = []

    # Single shared sub-sample index across the block so CH values
    # within the block are directly comparable.
    sub_rng = np.random.default_rng(args.surrogate_seed * 9973 + args.k)
    sub_size = min(args.ch_subsample, surrogate_cpu.shape[0])
    sub_idx = sub_rng.choice(surrogate_cpu.shape[0], sub_size, replace=False)
    sub_idx.sort()
    sub_x_cpu = np.asarray(surrogate_cpu[sub_idx], dtype=np.float32)

    for fit_idx in range(args.n_fits):
        kmeans_seed = args.kmeans_seed_base + fit_idx
        t0 = datetime.datetime.now()
        print(
            f"[{t0}] fit {fit_idx + 1}/{args.n_fits}  "
            f"(surrogate_seed={args.surrogate_seed}, k={args.k}, "
            f"kmeans_seed={kmeans_seed})",
        )

        kmeans = KMeans(
            init="k-means||",
            n_clusters=args.k,
            random_state=kmeans_seed,
        )
        kmeans.fit(surrogate_gpu)

        centroids = kmeans.cluster_centers_.get()
        # Sub-sampled CH — labels for the sub-sample re-derived by
        # nearest-centroid assignment so we never .get() the full
        # 2e8 label vector.
        c_cpu = centroids.astype(np.float32)
        c_norm = (c_cpu * c_cpu).sum(axis=1)
        scores = sub_x_cpu @ c_cpu.T
        sub_labels = (-2.0 * scores + c_norm).argmin(axis=1)

        # Full-data inertia is what cuML reports natively (W(k) =
        # within-cluster sum of squared distances).
        inertia = float(kmeans.inertia_) if hasattr(kmeans, "inertia_") else float("nan")

        if len(np.unique(sub_labels)) > 1:
            ch = float(calinski_harabasz_score(sub_x_cpu, sub_labels))
            # Silhouette is O(N^2) — use sklearn's sample_size kwarg to
            # keep the pairwise distance matrix bounded to ~64 MB
            # regardless of k.  random_state pinned to kmeans_seed so
            # the sample selection is reproducible per fit.
            sil = float(silhouette_score(
                sub_x_cpu, sub_labels,
                sample_size=min(4000, sub_x_cpu.shape[0]),
                random_state=kmeans_seed,
            ))
        else:
            ch = float("nan")
            sil = float("nan")

        dt = (datetime.datetime.now() - t0).total_seconds()
        print(
            f"  CH (sub) = {ch:.1f}, silhouette (sub) = {sil:.4f}, "
            f"inertia = {inertia:.3e}, fit time = {dt:.1f}s",
        )

        centroids_list.append(centroids)
        ch_scores.append(ch)
        inertias.append(inertia)
        silhouettes.append(sil)
        durations.append(dt)

    unscale_vec: np.ndarray | None = None
    if args.unscale_csv is not None:
        # pd is imported at module scope; don't shadow it here.
        unscale_vec = pd.read_csv(args.unscale_csv).iloc[0].to_numpy(
            dtype=np.float32,
        )
        print(
            f"[{datetime.datetime.now()}] Loaded per-column divisors "
            f"{unscale_vec} from {args.unscale_csv}",
        )

    print(f"[{datetime.datetime.now()}] Computing Hungarian centroid drift")
    drift = hungarian_centroid_drift(
        centroids_list, ch_scores=ch_scores, unscale=unscale_vec,
    )

    rows = [
        {
            "surrogate_seed":   args.surrogate_seed,
            "k":                args.k,
            "kmeans_seed":      args.kmeans_seed_base + fit_idx,
            "ch":               ch_scores[fit_idx],
            "silhouette":       silhouettes[fit_idx],
            "inertia":          inertias[fit_idx],
            "instability_null": drift[fit_idx],
            "fit_duration_s":   durations[fit_idx],
        }
        for fit_idx in range(args.n_fits)
    ]
    df = pd.DataFrame(rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[{datetime.datetime.now()}] Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
