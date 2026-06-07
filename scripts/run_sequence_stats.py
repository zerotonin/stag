#!/usr/bin/env python
# ╔═══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.run_sequence_stats                                ║
# ║  « Sprint 3 driver: super-prototypes + circadian + per-animal »   ║
# ╠═══════════════════════════════════════════════════════════════════╣
# ║  Produces every Sprint 3 deliverable in one shot:                 ║
# ║                                                                   ║
# ║    results/tables/super_prototype_triplets.csv                    ║
# ║      Top n-grams with empirical count, null median, q-value,      ║
# ║      super_prototype flag.  R3 Q5.                                ║
# ║                                                                   ║
# ║    results/tables/circadian_hourly_proportions.csv                ║
# ║      Per-hour PM proportions across the cohort.  Feeds the        ║
# ║    results/figures/circadian_diurnal.{svg,png}                    ║
# ║      panel with day-1 / day-2 overlay (R1 + R3).                  ║
# ║                                                                   ║
# ║    results/tables/ear_flick_day_night.csv                         ║
# ║      Per-animal rates + Wilcoxon test result + median ratio.      ║
# ║      R1 #10.                                                      ║
# ║                                                                   ║
# ║    results/tables/per_animal_time_budget.csv                      ║
# ║    results/figures/per_animal_time_budget.{svg,png}               ║
# ║      Stacked bar per stag, ordered by inactive proportion.        ║
# ║      R2 #8.                                                       ║
# ║                                                                   ║
# ║  Inputs (all on the local NVMe data drive):                       ║
# ║    - cluster_results/.../delSize_0/k_8/labels/*.npy (any one)     ║
# ║    - label_timeline_deer_ids.npy   (built by cache_label_timeline)║
# ║    - label_timeline_timestamps.npy (built by cache_label_timeline)║
# ╚═══════════════════════════════════════════════════════════════════╝
"""Sprint 3 driver: super-prototypes, circadian, ear-flick day/night, per-animal."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stag.analysis.circadian import (
    ear_flick_day_night_test,
    hourly_proportions,
    per_animal_time_budget,
    split_by_day,
)
from stag.analysis.super_prototypes import (
    per_animal_bout_streams,
)
from stag.constants import (
    CANONICAL_K8_LABELS,
    LABEL_TIMELINE_DEER_IDS,
    LABEL_TIMELINE_TIMESTAMPS,
    PM_CATEGORY,
    PM_COLOURS,
    PM_DISPLAY_NAMES,
    RESULTS_DIR_DEFAULT,
    apply_figure_defaults,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", type=Path, default=None,
        help="k=8 labels .npy (default: first match in "
             "cluster_results/.../delSize_0/k_8/labels/*.npy).",
    )
    parser.add_argument("--deer-ids",   type=Path, default=LABEL_TIMELINE_DEER_IDS)
    parser.add_argument("--timestamps", type=Path, default=LABEL_TIMELINE_TIMESTAMPS)
    parser.add_argument("--output-dir", type=Path,
                        default=RESULTS_DIR_DEFAULT / "sprint3")
    parser.add_argument("--n-shuffles", type=int, default=1000)
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--fdr-alpha",  type=float, default=0.05)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument(
        "--max-bouts", type=int, default=None,
        help="Optional cap on the bout sequence length used for the "
             "null model (random contiguous block).  Speeds up the "
             "1000-shuffle pass on the full cohort.  Default: no cap.",
    )
    return parser.parse_args()


def _resolve_labels(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if CANONICAL_K8_LABELS.exists():
        return CANONICAL_K8_LABELS
    raise SystemExit(
        f"Canonical k=8 labels {CANONICAL_K8_LABELS.name} not found.  "
        "Regenerate with the nearest-manuscript-centroid pipeline or "
        "pass --labels explicitly.",
    )


def _ear_flick_pms() -> list[int]:
    return [pm for pm, cat in PM_CATEGORY.items() if cat == "ear_flick"]


def _activity_pms() -> list[int]:
    return [pm for pm in PM_CATEGORY if pm not in (0,)]  # PM0 = quiescent excluded


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ─── Load aligned arrays ────────────────────────────────────────
    labels_path = _resolve_labels(args.labels)
    print(f"Loading labels:     {labels_path}")
    idx = np.load(labels_path)
    print(f"Loading deer_ids:   {args.deer_ids}")
    deer_ids = np.load(args.deer_ids,   mmap_mode="r")
    print(f"Loading timestamps: {args.timestamps}")
    timestamps_ns = np.load(args.timestamps, mmap_mode="r")
    print(f"  shapes: idx={idx.shape}  deer_ids={deer_ids.shape}  ts={timestamps_ns.shape}")
    if not (idx.shape == deer_ids.shape == timestamps_ns.shape):
        raise SystemExit(
            "Aligned-array shape mismatch.  Re-run cache_label_timeline.py?",
        )

    rng = np.random.default_rng(args.seed)
    pm_ids = sorted(set(PM_DISPLAY_NAMES.keys()))

    # ─── Super-prototype detection ──────────────────────────────────
    print()
    print("── Super-prototypes (bout-level triplets, first-order Markov null) ──")
    streams = per_animal_bout_streams(idx, np.asarray(deer_ids))
    bout_labels = np.concatenate([s.labels for s in streams.values()])
    print(f"  bout count (across {len(streams)} animals): {bout_labels.size:,}")

    if args.max_bouts is not None and bout_labels.size > args.max_bouts:
        start = rng.integers(0, bout_labels.size - args.max_bouts)
        bout_labels_test = bout_labels[start:start + args.max_bouts]
        print(f"  capping bout sequence at {args.max_bouts:,} (random contiguous slice)")
        print("  super-prototype test runs on the cap (both observed AND null) "
              "so the comparison is apples-to-apples")
    else:
        bout_labels_test = bout_labels

    # Both observed AND null must be computed on the same sequence so
    # the count comparison is consistent.  Using the cap for the null
    # but the full sequence for observed would inflate every "observed"
    # count by (full / cap) and produce a spurious "super-prototype"
    # call on essentially every triplet.
    from stag.analysis.null_models import (
        flag_significant_ngrams,
        ngram_frequencies,
        null_distribution,
    )
    n_states = max(int(bout_labels.max()) + 1, max(pm_ids) + 1)
    observed = ngram_frequencies(bout_labels_test, n=3, n_states=n_states)
    null = null_distribution(
        bout_labels_test, n=3, n_shuffles=args.n_shuffles,
        null_kind="first_order", rng=rng, n_states=n_states,
        desc="null shuffles",
    )
    triplets = flag_significant_ngrams(
        observed, null,
        percentile=args.percentile, fdr_alpha=args.fdr_alpha,
    )

    triplet_df = pd.DataFrame(triplets)
    triplet_df.insert(0, "ngram_str", triplet_df["ngram"].astype(str))
    triplet_csv = tables_dir / "super_prototype_triplets.csv"
    triplet_df.to_csv(triplet_csv, index=False)
    n_super = int(triplet_df["super_prototype"].sum())
    print(f"  wrote {triplet_csv}")
    print(f"  super-prototypes flagged (percentile AND FDR): {n_super}")
    print(triplet_df.head(10).to_string(index=False))

    # ─── Circadian hourly proportions + day-1/day-2 panel ───────────
    print()
    print("── Circadian: hourly PM proportions ──")
    day_idx = split_by_day(np.asarray(timestamps_ns), np.asarray(deer_ids))

    hp_all = hourly_proportions(idx, np.asarray(timestamps_ns), pm_ids=pm_ids)
    hp_all.to_csv(tables_dir / "circadian_hourly_proportions.csv")
    print(f"  wrote {tables_dir / 'circadian_hourly_proportions.csv'}")

    day_idx_arr = np.asarray(day_idx)
    panels = {}
    for d in (0, 1):
        mask = day_idx_arr == d
        if mask.sum() == 0:
            continue
        panels[d] = hourly_proportions(
            idx[mask], np.asarray(timestamps_ns)[mask], pm_ids=pm_ids,
        )

    _plot_diurnal(panels, hp_all, figures_dir)

    # ─── Ear-flick day/night test (R1 #10) ──────────────────────────
    print()
    print("── Ear-flick day/night Wilcoxon ──")
    ear_pms = _ear_flick_pms()
    act_pms = _activity_pms()
    test = ear_flick_day_night_test(
        idx, np.asarray(timestamps_ns), np.asarray(deer_ids),
        ear_flick_pms=ear_pms, activity_pms=act_pms,
    )
    test["per_animal"].to_csv(tables_dir / "ear_flick_day_night.csv")
    print(f"  wrote {tables_dir / 'ear_flick_day_night.csv'}")
    print(f"  n animals in test: {test['n_animals_in_test']}")
    print(f"  W = {test['W']}, p = {test['p_value']:.4g}")
    print(f"  median day/night rate ratio: {test['median_ratio_day_over_night']:.3f}")

    # ─── Per-animal time budget (R2 #8) ─────────────────────────────
    print()
    print("── Per-animal time budget ──")
    tb = per_animal_time_budget(idx, np.asarray(deer_ids), pm_ids=pm_ids)
    tb.to_csv(tables_dir / "per_animal_time_budget.csv")
    print(f"  wrote {tables_dir / 'per_animal_time_budget.csv'}")
    _plot_per_animal_time_budget(tb, pm_ids, figures_dir)

    print()
    print("Sprint 3 driver: done.")


def _plot_diurnal(
    panels: dict[int, pd.DataFrame],
    hp_all: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """One row per PM, with day-1 and day-2 overlaid; cohort line dashed."""
    apply_figure_defaults()
    pm_ids = [c for c in hp_all.columns if c != "n_samples"]
    n_pm = len(pm_ids)
    fig, axes = plt.subplots(n_pm, 1, sharex=True, figsize=(6.0, 1.2 * n_pm))
    if n_pm == 1:
        axes = [axes]
    for ax, pm in zip(axes, pm_ids):
        for d in sorted(panels):
            ax.plot(panels[d].index, panels[d][pm], marker="o", markersize=3,
                    linewidth=1.0, label=f"day {d+1}",
                    color=PM_COLOURS.get(int(pm), "#444444"),
                    alpha=0.4 + 0.3 * d)
        ax.plot(hp_all.index, hp_all[pm], color="#222222", linestyle="--",
                linewidth=0.8, label="cohort")
        ax.set_ylabel(f"PM{pm}\n{PM_DISPLAY_NAMES.get(int(pm), pm)}", fontsize=8)
        ax.set_ylim(0, max(0.05, hp_all[pm].max() * 1.2))
    axes[-1].set_xlabel("Hour of day (NZDT)")
    axes[0].legend(loc="upper right", frameon=False, fontsize="x-small")
    fig.tight_layout()
    save_figure(fig, "circadian_diurnal", figures_dir, data=hp_all.reset_index())


def _plot_per_animal_time_budget(
    tb: pd.DataFrame, pm_ids: list[int], figures_dir: Path,
) -> None:
    apply_figure_defaults()
    pm_cols = [c for c in tb.columns if c != "n_samples"]
    inactive_share = tb[[c for c in pm_cols if PM_CATEGORY.get(int(c)) == "inactive"]].sum(axis=1)
    order = inactive_share.sort_values(ascending=False).index
    tb_sorted = tb.loc[order, pm_cols]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    bottom = np.zeros(len(tb_sorted))
    for pm in pm_cols:
        ax.bar(
            [str(d) for d in tb_sorted.index],
            tb_sorted[pm].values, bottom=bottom,
            color=PM_COLOURS.get(int(pm), "#888888"),
            edgecolor="white", linewidth=0.3,
            label=f"PM{pm}: {PM_DISPLAY_NAMES.get(int(pm), pm)}",
        )
        bottom = bottom + tb_sorted[pm].values
    ax.set_xlabel("Animal (deer_id, ordered by inactive proportion)")
    ax.set_ylabel("Proportion of time")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, frameon=False, fontsize="x-small")
    fig.tight_layout()
    save_figure(fig, "per_animal_time_budget", figures_dir,
                data=tb_sorted.reset_index())


if __name__ == "__main__":
    main()
