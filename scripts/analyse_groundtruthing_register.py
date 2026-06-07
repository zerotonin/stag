#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.analyse_groundtruthing_register                  ║
# ║  « token-overlap external validation when labels.npy is absent » ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Parses the free-text "Behaviour" column of a DINZ groundtruthing║
# ║  register into a canonical behaviour-token vocabulary, computes  ║
# ║  the row-normalised cluster x token confusion matrix, the per-   ║
# ║  cluster prototype-agreement fraction with Wilson 95% CIs, and   ║
# ║  writes one PNG + SVG + CSV trio per artefact.                   ║
# ║                                                                  ║
# ║  This is the canonical external-validation tool for STAG.  We   ║
# ║  do not run Hungarian-aligned ARI / NMI / V-measure against the  ║
# ║  current labels.npy because the registers' Cluster Label IDs    ║
# ║  were authored against earlier k-means fits and the pair-based  ║
# ║  metrics would mis-attribute partition drift to disagreement.   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Token-overlap external validation of a groundtruthing register."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stag.constants import (
    PM_DISPLAY_NAMES,
    RESULTS_DIR_DEFAULT,
    apply_figure_defaults,
    save_figure,
)


# ┌────────────────────────────────────────────────────────────┐
# │ Behaviour-token vocabulary  « free-text -> canonical set » │
# └────────────────────────────────────────────────────────────┘

TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "not_in_view": re.compile(r"not[ _]?in[ _]?view|notinview|no in view", re.I),
    "ear_flick":   re.compile(r"earfl|earflick|ear flick|ear[_ ]?fl\b|earswing|ear swing", re.I),
    "graze":       re.compile(r"\bgraze|\bgr\b|grazing|gr\+|gr ", re.I),
    "ruminate":    re.compile(r"(?:^|[ _])rum(?:[ _]|$)|ruminat|lyrum", re.I),
    "pant":        re.compile(r"\bpant", re.I),
    "walk":        re.compile(r"\bwalk|\bwalkg|stepg|steps?|step@", re.I),
    "lying":       re.compile(r"\bly\b|\bly_|\blying|lyrum|lypant", re.I),
    "standing":    re.compile(r"\bstand|\bst_|standstill", re.I),
    "head_still":  re.compile(r"headstill|head[_ ]still|hdstill|hd[_ ]still", re.I),
    "head_move":   re.compile(
        r"head[_ ]?turn|hdturn|headmove|hd[_ ]?move|head ?up|hdup|"
        r"headdown|hd[_ ]?down|head ?moving|hdmoving|hd ?swing|"
        r"head ?shake|headshake|hdbob|head ?bob",
        re.I,
    ),
    "ear_still":   re.compile(r"ear[s]?still|earstill|ears[_ ]still|ear[_ ]still", re.I),
    "ear_move":    re.compile(r"earmvt|ear ?mv|earout|earsback|earsfwd|earmove|ear ?move|ear[_ ]?turn", re.I),
    "groom":       re.compile(r"groom|grooming|rubnose|rub[_ ]?nose|footscratch|foot ?scratch", re.I),
    "drink":       re.compile(r"\bdrink", re.I),
    "yawn":        re.compile(r"\byawn|yawing", re.I),
}

# Order tokens read left-to-right in the confusion matrix: occlusion,
# then the immobile/active body-axis tokens, then ear axis, then mouth
# behaviours.  Keeps related columns adjacent.
TOKEN_ORDER: tuple[str, ...] = (
    "not_in_view",
    "lying", "standing", "head_still", "head_move",
    "ear_still", "ear_move", "ear_flick",
    "graze", "walk",
    "pant", "ruminate",
    "groom", "drink", "yawn",
)

CLUSTER_EXPECT: dict[int, set[str]] = {
    0: {"lying", "standing", "head_still", "ear_still"},
    1: {"lying", "head_still", "ear_still", "ruminate", "pant"},
    2: {"ear_flick", "ear_move", "head_move", "walk", "standing"},
    3: {"lying", "standing", "head_still", "ear_still", "pant", "ruminate"},
    4: {"ear_flick", "walk", "graze", "groom", "head_move", "pant"},
    5: {"ear_flick", "walk", "graze", "groom"},
    6: {"graze", "walk"},
    7: {"graze"},
}


# ┌────────────────────────────────────────────────────────────┐
# │ Parsing  « free-text behaviour -> token set »              │
# └────────────────────────────────────────────────────────────┘

def tokenise(text: str) -> set[str]:
    """Return the set of behaviour tokens that match ``text``."""
    return {name for name, rgx in TOKEN_PATTERNS.items() if rgx.search(text)}


def load_register(csv_path: Path) -> pd.DataFrame:
    """Read a register CSV and add tokens / agreement columns."""
    df = pd.read_csv(csv_path)
    df["Behaviour"] = df["Behaviour"].fillna("").astype(str)
    df["Cluster Label"] = df["Cluster Label"].astype(int)
    df["tokens"] = df["Behaviour"].apply(tokenise)
    df["not_in_view"] = df["tokens"].apply(lambda s: "not_in_view" in s)
    df["agrees"] = df.apply(
        lambda r: bool(r["tokens"] & CLUSTER_EXPECT[int(r["Cluster Label"])]),
        axis=1,
    )
    return df


# ┌────────────────────────────────────────────────────────────┐
# │ Confusion matrix  « row-normalised cluster x token »       │
# └────────────────────────────────────────────────────────────┘

def build_token_confusion(
    df: pd.DataFrame, exclude_oov: bool = False,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Return (matrix, cluster_ids, token_names) row-normalised by cluster size.

    When ``exclude_oov`` is true, ``not_in_view`` windows are dropped
    from the denominator and the ``not_in_view`` token is removed from
    the column axis — yielding the in-view-only signature.
    """
    if exclude_oov:
        df = df[~df["not_in_view"]]
        tokens = [t for t in TOKEN_ORDER if t != "not_in_view"]
    else:
        tokens = list(TOKEN_ORDER)
    clusters = sorted(df["Cluster Label"].unique().tolist())
    M = np.zeros((len(clusters), len(tokens)), dtype=float)
    for i, k in enumerate(clusters):
        sub = df[df["Cluster Label"] == k]
        n = len(sub)
        ctr: Counter[str] = Counter()
        for s in sub["tokens"]:
            ctr.update(s)
        for j, t in enumerate(tokens):
            M[i, j] = ctr.get(t, 0) / n if n else 0.0
    return M, clusters, tokens


# ┌────────────────────────────────────────────────────────────┐
# │ PM x PM confusion  « video tokens -> best-matching PM »    │
# └────────────────────────────────────────────────────────────┘

def video_pm_distribution(
    tokens: set[str], assigned_pm: int,
) -> dict[int, float]:
    """Conditional decision rule for PM attribution.

    Step 1 — does the observer's description fit the assigned PM?
    If ``tokens`` overlaps ``CLUSTER_EXPECT[assigned_pm]`` at all,
    the window is counted as the assigned PM (full weight on the
    diagonal).

    Step 2 — only when the description does NOT fit the assigned PM
    do we ask which other PM it would fit instead.  We then take the
    Jaccard winner over the other seven PMs' expected sets, splitting
    the unit weight evenly across ties.

    Returns ``{}`` when ``tokens`` has no overlap with any PM (window
    should be excluded from the confusion matrix).
    """
    observed = tokens - {"not_in_view"}
    if not observed:
        return {}

    # Step 1: fits the assigned PM -> stay on the diagonal.
    if observed & CLUSTER_EXPECT[assigned_pm]:
        return {assigned_pm: 1.0}

    # Step 2: best Jaccard match among the OTHER seven PMs.
    scores: dict[int, float] = {}
    for k, expected in CLUSTER_EXPECT.items():
        if k == assigned_pm:
            continue
        inter = observed & expected
        if not inter:
            scores[k] = 0.0
            continue
        union = observed | expected
        scores[k] = len(inter) / len(union)
    if not scores or max(scores.values()) == 0.0:
        return {}
    top = max(scores.values())
    winners = [k for k, s in scores.items() if s == top]
    return {k: 1.0 / len(winners) for k in winners}


def build_pm_confusion(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[int], dict[str, object]]:
    """Row-normalised PM x PM confusion matrix on in-view windows only.

    Rows: the cluster the window was assigned to (``Cluster Label``).
    Cols: the PM whose ``CLUSTER_EXPECT`` set best matches the
          observer's tokens (Jaccard winner; ties split evenly).
    Off-diagonal mass measures how often a window's video description
    was more consistent with a different prototype's expected tokens.
    """
    evaluable = df[~df["not_in_view"]]
    clusters = sorted(CLUSTER_EXPECT.keys())
    n_pm = len(clusters)
    counts = np.zeros((n_pm, n_pm), dtype=float)
    n_no_overlap = 0
    n_used = 0
    cluster_to_idx = {k: i for i, k in enumerate(clusters)}

    for _, row in evaluable.iterrows():
        assigned = int(row["Cluster Label"])
        dist = video_pm_distribution(row["tokens"], assigned)
        if not dist:
            n_no_overlap += 1
            continue
        i = cluster_to_idx[assigned]
        for k, w in dist.items():
            counts[i, cluster_to_idx[k]] += w
        n_used += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    M_norm = counts / row_sums

    info: dict[str, object] = {
        "n_in_view": int(len(evaluable)),
        "n_used": int(n_used),
        "n_no_overlap": int(n_no_overlap),
        "raw_counts": counts,
        "diagonal_mean": float(np.diag(M_norm).mean()),
        "diagonal_per_pm": {k: float(M_norm[i, i]) for k, i in cluster_to_idx.items()},
    }
    return M_norm, clusters, info


# ┌────────────────────────────────────────────────────────────┐
# │ Wilson 95% CI  « binomial proportion confidence interval » │
# └────────────────────────────────────────────────────────────┘

def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the Wilson score 95% CI for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_cluster_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster agreement on evaluable rows with Wilson 95% CIs."""
    rows = []
    for k in sorted(df["Cluster Label"].unique().tolist()):
        sub = df[df["Cluster Label"] == k]
        n_total = len(sub)
        n_view = int((~sub["not_in_view"]).sum())
        evaluable = sub[~sub["not_in_view"]]
        n_agree = int(evaluable["agrees"].sum())
        p = n_agree / n_view if n_view else 0.0
        lo, hi = wilson_ci(n_agree, n_view)
        rows.append({
            "k": k,
            "name": PM_DISPLAY_NAMES.get(k, str(k)),
            "n_total": n_total,
            "n_not_in_view": int(sub["not_in_view"].sum()),
            "n_evaluable": n_view,
            "n_agree": n_agree,
            "agreement": p,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    return pd.DataFrame(rows)


# ┌────────────────────────────────────────────────────────────┐
# │ Figure  « cluster x behaviour-token heatmap »              │
# └────────────────────────────────────────────────────────────┘

def plot_pm_confusion(
    M: np.ndarray,
    clusters: list[int],
    info: dict[str, object],
) -> plt.Figure:
    """Render the row-normalised PM x PM external confusion matrix."""
    apply_figure_defaults()
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    labels = [f"PM{k}\n{PM_DISPLAY_NAMES.get(k, '')}" for k in clusters]
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize="small")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels(labels, fontsize="small")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v >= 0.02:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.5 else "black")

    ax.set_xlabel("Video-inferred PM (assigned PM if description fits, else best other-PM Jaccard match)")
    ax.set_ylabel("Cluster-assigned PM (k = 8)")
    ax.set_title(
        f"PM x PM external confusion (in-view, conditional decision rule) — "
        f"{int(info['n_used'])} of {int(info['n_in_view'])} windows used, "
        f"diagonal mean = {float(info['diagonal_mean']):.2f}",
        fontsize="small",
    )
    fig.colorbar(im, ax=ax, label="row-normalised fraction")
    fig.tight_layout()
    return fig


def plot_confusion(
    M: np.ndarray,
    clusters: list[int],
    tokens: list[str],
    agreement: pd.DataFrame,
    title_extra: str = "",
) -> plt.Figure:
    """Render the row-normalised cluster x token confusion matrix."""
    apply_figure_defaults()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(
        [t.replace("_", " ") for t in tokens],
        rotation=45, ha="right", fontsize="small",
    )

    ytick_labels = [
        f"PM{k} {PM_DISPLAY_NAMES.get(k, '')}\n"
        f"(n={int(row.n_evaluable)}, agree={row.agreement:.0%})"
        for k, (_, row) in zip(clusters, agreement.iterrows())
    ]
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels(ytick_labels, fontsize="small")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.5 else "black")

    ax.set_xlabel("Observed behaviour token (free-text video annotation)")
    ax.set_ylabel("Prototypical movement (k = 8)")
    overall = agreement["n_agree"].sum() / agreement["n_evaluable"].sum()
    ax.set_title(
        f"Token-overlap external validation{title_extra} — "
        f"{int(agreement['n_total'].sum())} windows, "
        f"{int(agreement['n_evaluable'].sum())} evaluable, "
        f"overall agreement {overall:.0%}",
        fontsize="small",
    )
    fig.colorbar(im, ax=ax, label="fraction of windows in PM (row-normalised)")
    fig.tight_layout()
    return fig


def plot_agreement_bars(agreement: pd.DataFrame) -> plt.Figure:
    """Horizontal agreement bar chart with Wilson 95% CIs."""
    apply_figure_defaults()
    from stag.constants import PM_COLOURS
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    y = np.arange(len(agreement))[::-1]
    p = agreement["agreement"].values
    lo = np.clip(p - agreement["ci_lo"].values, 0.0, None)
    hi = np.clip(agreement["ci_hi"].values - p, 0.0, None)
    colours = [PM_COLOURS[int(k)] for k in agreement["k"]]

    ax.barh(y, p, xerr=[lo, hi], color=colours, edgecolor="black",
            linewidth=0.5, error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2.5})
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"PM{int(r['k'])} {r['name']}\n"
         f"{int(r['n_agree'])}/{int(r['n_evaluable'])} evaluable "
         f"({int(r['n_not_in_view'])} not-in-view)"
         for _, r in agreement.iterrows()],
        fontsize="small",
    )
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Prototype agreement (token overlap) — Wilson 95% CI")
    overall = agreement["n_agree"].sum() / agreement["n_evaluable"].sum()
    ax.set_title(
        f"Per-PM external agreement — overall {overall:.1%} "
        f"({int(agreement['n_agree'].sum())} / "
        f"{int(agreement['n_evaluable'].sum())} evaluable windows)",
        fontsize="small",
    )
    fig.tight_layout()
    return fig


# ┌────────────────────────────────────────────────────────────┐
# │ CLI  « parse args, load, compute, save »                   │
# └────────────────────────────────────────────────────────────┘

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "register", type=Path,
        help="Path to a DINZ groundtruthing register CSV.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULTS_DIR_DEFAULT / "sprint2" / "qualitative",
        help="Where to write figures + tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading register: {args.register}")
    df = load_register(args.register)
    print(f"  {len(df)} rows, "
          f"{int(df['not_in_view'].sum())} not-in-view, "
          f"{int((~df['not_in_view']).sum())} evaluable")

    agreement = per_cluster_agreement(df)
    print("\nPer-PM agreement (Wilson 95% CI):")
    print(agreement.to_string(index=False, float_format="%.3f"))

    overall = agreement["n_agree"].sum() / agreement["n_evaluable"].sum()
    print(f"\nOverall: {overall:.1%} "
          f"({int(agreement['n_agree'].sum())} / "
          f"{int(agreement['n_evaluable'].sum())} evaluable)")

    # ─── Token confusion (all 160 rows, includes OOV column) ─────────
    M_all, clusters_all, tokens_all = build_token_confusion(df, exclude_oov=False)
    M_all_df = pd.DataFrame(
        M_all, index=[f"PM{k}" for k in clusters_all], columns=tokens_all,
    )
    M_all_long = (M_all_df.reset_index().melt(id_vars="index", var_name="token",
                                              value_name="row_normalised_fraction")
                  .rename(columns={"index": "cluster"}))
    fig1 = plot_confusion(M_all, clusters_all, tokens_all, agreement)
    save_figure(fig1, "figure_groundtruth_token_confusion",
                figures_dir, data=M_all_long)

    # ─── Token confusion (in-view only — OOV row & column dropped) ───
    M_iv, clusters_iv, tokens_iv = build_token_confusion(df, exclude_oov=True)
    iv_agreement = (agreement.assign(
        n_total=agreement["n_evaluable"]  # denominator becomes in-view count
    ))
    M_iv_df = pd.DataFrame(
        M_iv, index=[f"PM{k}" for k in clusters_iv], columns=tokens_iv,
    )
    M_iv_long = (M_iv_df.reset_index().melt(id_vars="index", var_name="token",
                                            value_name="row_normalised_fraction")
                 .rename(columns={"index": "cluster"}))
    fig1b = plot_confusion(M_iv, clusters_iv, tokens_iv, iv_agreement,
                           title_extra=" (in-view only)")
    save_figure(fig1b, "figure_groundtruth_token_confusion_inview",
                figures_dir, data=M_iv_long)

    # ─── PM x PM confusion (in-view only, Jaccard token-set match) ───
    M_pm, clusters_pm, info_pm = build_pm_confusion(df)
    print(
        f"\nPM x PM confusion (in-view only): "
        f"{info_pm['n_used']} / {info_pm['n_in_view']} windows used "
        f"({info_pm['n_no_overlap']} had no token overlap with any PM); "
        f"diagonal mean = {float(info_pm['diagonal_mean']):.3f}"
    )
    diag = info_pm["diagonal_per_pm"]
    print("Per-PM diagonal (self-attribution under Jaccard):")
    for k in clusters_pm:
        print(f"  PM{k} {PM_DISPLAY_NAMES.get(k, ''):>22}: {diag[k]:.2f}")

    M_pm_df = pd.DataFrame(
        M_pm,
        index=[f"PM{k}_assigned" for k in clusters_pm],
        columns=[f"PM{k}_video" for k in clusters_pm],
    )
    M_pm_long = (M_pm_df.reset_index().melt(id_vars="index", var_name="video_pm",
                                            value_name="row_normalised_fraction")
                 .rename(columns={"index": "assigned_pm"}))
    fig3 = plot_pm_confusion(M_pm, clusters_pm, info_pm)
    save_figure(fig3, "figure_groundtruth_pm_vs_pm_confusion",
                figures_dir, data=M_pm_long)

    # ─── Agreement bar chart (with companion CSV) ────────────────────
    fig2 = plot_agreement_bars(agreement)
    save_figure(fig2, "figure_groundtruth_per_pm_agreement",
                figures_dir, data=agreement)

    # ─── Tables ──────────────────────────────────────────────────────
    agreement.to_csv(tables_dir / "table_per_pm_agreement.csv", index=False)
    M_all_df.to_csv(tables_dir / "table_token_confusion_matrix.csv")
    M_iv_df.to_csv(tables_dir / "table_token_confusion_matrix_inview.csv")
    M_pm_df.to_csv(tables_dir / "table_pm_vs_pm_confusion.csv")
    pd.DataFrame(info_pm["raw_counts"],
                 index=[f"PM{k}_assigned" for k in clusters_pm],
                 columns=[f"PM{k}_video" for k in clusters_pm]
                ).to_csv(tables_dir / "table_pm_vs_pm_confusion_counts.csv")

    print(f"\nWrote:")
    for p in (figures_dir / "figure_groundtruth_token_confusion.png",
              figures_dir / "figure_groundtruth_token_confusion_inview.png",
              figures_dir / "figure_groundtruth_pm_vs_pm_confusion.png",
              figures_dir / "figure_groundtruth_per_pm_agreement.png",
              tables_dir / "table_per_pm_agreement.csv",
              tables_dir / "table_token_confusion_matrix.csv",
              tables_dir / "table_token_confusion_matrix_inview.csv",
              tables_dir / "table_pm_vs_pm_confusion.csv",
              tables_dir / "table_pm_vs_pm_confusion_counts.csv"):
        print(f"  {p}")


if __name__ == "__main__":
    main()
