"""
Radar-chart and heatmap visualisation of cluster centroids.

Provides :class:`ClusterPlotter` for generating per-centroid radar
charts from a JSON centroids file.
"""
import matplotlib.pyplot as plt
import numpy as np
import json

class ClusterPlotter:
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

# Usage
plotter = ClusterPlotter('/home/geuba03p/deer_cluster/centroid_label_info.json')
plotter.plot_radar_and_metrics('nmax')
