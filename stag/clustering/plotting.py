# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — clustering.plotting                                      ║
# ║  « per-centroid dashboards and internal-metric figures »         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  CentroidDashboard renders the per-PM radar + pie + bar panel   ║
# ║  used in Figure 3 of the manuscript.                            ║
# ║                                                                  ║
# ║  Free functions below produce the internal-metric figures       ║
# ║  (Calinski–Harabasz, Silhouette, Inertia/Kneedle, stability)    ║
# ║  that make up the revised Figure 2.                              ║
# ║                                                                  ║
# ║  Cross-run lineplots that aggregate over the meta-analysis      ║
# ║  DataFrame live in clustering.meta_analysis.ClusterPlotter.     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Per-centroid dashboards and internal-metric figures."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from stag.constants import WONG, apply_figure_defaults

if TYPE_CHECKING:
    import pandas as pd

class CentroidDashboard:
    def __init__(self, centroids_info_path):
        """Load centroids information and additional metrics from a JSON file."""
        with open(centroids_info_path, 'r') as file:
            data = json.load(file)
        self.centroids = data['centroids']
        self.feature_labels = data['feature_labels']

    def plot_radar_and_metrics(self,feature_set):
        """Plots a radar chart for each centroid with additional metrics."""
        n_clusters = len(self.centroids)
        
        # Setup figure and grid
        fig, axs = plt.subplots(nrows=3,ncols=3, figsize=(15, 10), subplot_kw=dict(polar=True))
        axs = axs.flatten() 
        if n_clusters == 1:
            axs = [axs]  # Ensure axs is iterable for a single cluster
        
        for ax, centroid_info in zip(axs[0:n_clusters], self.centroids):
            self._plot_single_cluster(ax, centroid_info,feature_set)
        
        plt.tight_layout()
        plt.show()

    def _plot_radar_chart(self, ax, centroid_info,feature_set):
        """Plot the radar chart for a single cluster."""
        angles = np.linspace(0, 2 * np.pi, len(self.feature_labels), endpoint=False).tolist() + [0]
        stats = np.array(centroid_info[f'feature_val_{feature_set}'] + [centroid_info[f'feature_val_{feature_set}'][0]])
        ax.plot(angles, stats, linewidth=2, linestyle='solid')
        ax.fill(angles, stats, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.feature_labels)

    def _plot_pie_chart(self, ax, centroid_info):
        """Create inset for Pie Chart and add percentage text below it."""
        ax_inset = ax.inset_axes([-0.5, 0.3, 0.5, 0.7])
        ax_inset.pie([centroid_info['percentage'], 100 - centroid_info['percentage']], startangle=90, counterclock=False, colors=['#ff9999','#66b3ff'])
        ax_inset.set_aspect("equal")
        percentage_text = f"{centroid_info['percentage']:.2f}%"
        ax_inset.text(0.5, -0.1, percentage_text, transform=ax_inset.transAxes, ha="center", va="top", fontsize=9)

    def _plot_bar_plots(self, ax, centroid_info):
        """Plot Horizontal Bar Plots for Tortuosity and Speed."""
        ax_bar = ax.inset_axes([-0.5, -0.5, 0.5, 0.5])
        bars_positions = np.arange(len(['Tortuosity', 'Abs Speed']))
        values = [centroid_info['tortuosity_mean'], centroid_info['abs_speed_mPs_mean']]
        yerr = [centroid_info['tortuosity_sem'], centroid_info['abs_speed_mPs_sem']]
        ax_bar.barh(bars_positions, values, xerr=yerr, color=['#4CAF50', '#2196F3'])
        ax_bar.axis('off')

    def _plot_duration_text(self, ax, centroid_info):
        """Add duration text below the radar plot."""
        duration_text = f"{centroid_info['duration_sec_mean']:.2f} ± {centroid_info['duration_sec_sem']:.2f} sec"
        ax.text(0.5, -0.2, duration_text, transform=ax.transAxes, ha="center", va="top", fontsize=9)

    def _plot_single_cluster(self, ax, centroid_info,feature_set):
        """Plot a single cluster's radar chart and metrics using modularized methods."""
        self._plot_radar_chart(ax, centroid_info,feature_set)
        self._plot_pie_chart(ax, centroid_info)
        self._plot_bar_plots(ax, centroid_info)
        self._plot_duration_text(ax, centroid_info)

# ┌────────────────────────────────────────────────────────────┐
# │ Internal-metric figures  « Figure 2 of the revision »      │
# └────────────────────────────────────────────────────────────┘


def plot_internal_metrics_panel(
    summary: "pd.DataFrame",
    elbow_k: int | None = None,
    chosen_k: int | None = 8,
    figsize: tuple[float, float] = (9.0, 7.2),
) -> plt.Figure:
    """Four-panel internal-metric figure for the revised Figure 2.

    Panels:
      (A) Calinski–Harabasz index vs k.
      (B) Instability (Hungarian-matched centroid drift) vs k.
      (C) Mean stratified Silhouette vs k.
      (D) Inertia W(k) vs k with the Kneedle elbow marked.

    Args:
        summary:  Per-k DataFrame from
                  :func:`stag.clustering.internal_metrics.selection_summary`.
                  Columns: ``k``, ``calinski_harabasz``, ``instability``,
                  ``silhouette``, ``inertia``.
        elbow_k:  k flagged by the Kneedle algorithm (drawn as a marker
                  on panel D).  None to suppress.
        chosen_k: k highlighted across all panels as the manuscript's
                  selected solution.  Default 8.
        figsize:  Figure size in inches.

    Returns:
        The figure handle (caller saves via
        :func:`stag.constants.save_figure`).
    """
    apply_figure_defaults()

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    ax_ch, ax_inst, ax_sil, ax_W = axes.flatten()

    k = summary["k"].to_numpy()
    line_colour = WONG["blue"]
    chosen_colour = WONG["vermilion"]
    elbow_colour = WONG["orange"]

    def _band(name: str):
        """Return (low, high) columns for ``name`` if present, else None."""
        lo, hi = f"{name}_low", f"{name}_high"
        if lo in summary.columns and hi in summary.columns:
            return summary[lo], summary[hi]
        return None, None

    _plot_metric(ax_ch, k, summary["calinski_harabasz"],
                 *_band("calinski_harabasz"),
                 label="Calinski–Harabasz", colour=line_colour)
    ax_ch.set_title("(A) Quality")
    ax_ch.set_ylabel("CH index (higher is better)")

    _plot_metric(ax_inst, k, summary["instability"],
                 *_band("instability"),
                 label="Instability", colour=line_colour)
    ax_inst.set_title("(B) Stability")
    ax_inst.set_ylabel("Hungarian-matched drift\n(lower is better)")

    _plot_metric(ax_sil, k, summary["silhouette"],
                 *_band("silhouette"),
                 label="Silhouette", colour=line_colour)
    ax_sil.set_title("(C) Silhouette")
    ax_sil.set_ylabel("Mean silhouette ($\\bar{s}$)")
    ax_sil.set_xlabel("k")

    _plot_metric(ax_W, k, summary["inertia"],
                 *_band("inertia"),
                 label="Inertia", colour=line_colour)
    ax_W.set_title("(D) Inertia / Elbow")
    ax_W.set_ylabel("$W(k)$")
    ax_W.set_xlabel("k")

    if elbow_k is not None and elbow_k in k:
        y_at_elbow = summary.loc[summary["k"] == elbow_k, "inertia"].iloc[0]
        ax_W.scatter(
            [elbow_k], [y_at_elbow],
            s=80, color=elbow_colour, zorder=5,
            label=f"Kneedle elbow (k = {elbow_k})",
        )
        ax_W.legend(loc="upper right", frameon=False, fontsize="small")

    if chosen_k is not None:
        for ax in (ax_ch, ax_inst, ax_sil, ax_W):
            ax.axvline(chosen_k, color=chosen_colour, linestyle="--",
                       linewidth=1.0, alpha=0.7)

    # Show ticks at the actual k values (which are non-uniformly
    # spaced past k=20: 2..20 step 1, then 25, 30, 35, 40, 45, 50).
    # The earlier `k[::2]` heuristic was visually confusing because
    # it skipped 25, 35, 45 while keeping 30, 40, 50.  Now: tick at
    # every k; minor ticks suppressed; labels rotated when crowded.
    for ax in (ax_ch, ax_inst, ax_sil, ax_W):
        ax.set_xticks(k)
        ax.set_xticks([], minor=True)
        if len(k) > 15:
            ax.tick_params(axis="x", labelsize="x-small", rotation=45)

    fig.tight_layout()
    return fig


def _plot_metric(
    ax, x, y, low=None, high=None,
    *, label: str, colour: str,
) -> None:
    """Single-panel helper — median line + IQR shading + markers.

    ``low`` and ``high`` are optional 1-D sequences of the same length
    as ``y`` (typically the 25th and 75th percentiles).  When both are
    provided, the panel renders a translucent fill between them;
    otherwise just the median line.  Matches the manuscript's
    Figure 2A "median ± IQR" convention.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(y_arr)

    if low is not None and high is not None:
        low_arr = np.asarray(low, dtype=float)
        high_arr = np.asarray(high, dtype=float)
        band_good = finite & np.isfinite(low_arr) & np.isfinite(high_arr)
        if band_good.any():
            ax.fill_between(
                x_arr[band_good], low_arr[band_good], high_arr[band_good],
                facecolor=colour, alpha=0.35,
                edgecolor=colour, linewidth=0.6,
                label=f"{label} IQR",
            )

    ax.plot(x_arr[finite], y_arr[finite],
            color=colour, marker="o", markersize=4, linewidth=1.5,
            label=label)


if __name__ == "__main__":
    plotter = CentroidDashboard("/home/geuba03p/deer_cluster/centroid_label_info.json")
    plotter.plot_radar_and_metrics("nmax")
