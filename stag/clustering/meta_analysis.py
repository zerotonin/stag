# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — clustering.meta_analysis                                 ║
# ║  « Hungarian-matched centroid stability »                        ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  ClusterMetaAnalysis loads the per-run metadata JSONs produced  ║
# ║  by stag.clustering.kmeans, applies the Hungarian assignment    ║
# ║  to pair-wise centroid distances, and identifies the most       ║
# ║  stable solution for each (k, reduction_percent) combination.   ║
# ║                                                                  ║
# ║  ClusterPlotter renders the resulting quality / instability     ║
# ║  panels and radar-chart views of individual centroid sets.      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Hungarian-matched centroid stability and meta-analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment


class ClusterMetaAnalysis:
    """Load clustering metadata and quantify centroid stability across runs.

    Walks a directory of per-run metadata JSONs (one per k-means fit)
    and computes the Hungarian-matched centroid instability for every
    ``(k, reduction_percent)`` combination.  The minimum-instability run
    in each group serves as the representative solution.

    Attributes:
        directory: Root directory containing the metadata JSON files.
        df:        DataFrame of per-run metadata + instability values
                   (populated by :meth:`analyze` or :meth:`load_df`).
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = str(directory)
        self.df: pd.DataFrame | None = None

    # ─────────────────────────────────────────────────────────────────
    #  Loading
    # ─────────────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        """Walk ``self.directory`` and return one row per metadata JSON.

        Returns:
            DataFrame with the JSON fields plus ``k_number`` (inferred
            from the centroid count) and ``file_path``.  The ``centroids``
            field is dropped to keep memory usage small.
        """
        data: list[dict] = []
        for root, _dirs, files in os.walk(self.directory):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                file_path = os.path.join(root, fname)
                try:
                    with open(file_path, "r") as f:
                        content = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    print(f"file not loadable: {file_path} ({e})")
                    continue

                content["k_number"] = len(content["centroids"])
                content.pop("centroids", None)
                content["file_path"] = file_path
                data.append(content)

        return pd.DataFrame(data)

    def load_centroids_for_analysis(
        self, reduction_percent: float, k_number: int,
    ) -> list[np.ndarray]:
        """Return a list of centroid arrays for one ``(reduction_percent, k)`` group.

        Each entry is a ``(k, n_features)`` array re-loaded from disk.
        """
        if self.df is None:
            raise RuntimeError("Call analyze() or load_df() before loading centroids.")

        filtered = self.df[
            (self.df["reduction_percent"] == reduction_percent)
            & (self.df["k_number"] == k_number)
        ]

        centroids_list: list[np.ndarray] = []
        for _, row in filtered.iterrows():
            with open(row["file_path"], "r") as f:
                content = json.load(f)
            centroids_list.append(np.array(content["centroids"]))

        return centroids_list

    # ─────────────────────────────────────────────────────────────────
    #  Instability via Hungarian assignment
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_instability(centroids_list: list[np.ndarray]) -> np.ndarray:
        """Pair-wise Hungarian-matched distances, ranked against the most-stable run.

        For each pair of clustering attempts, builds the per-centroid
        Euclidean distance matrix, solves the linear-sum assignment
        problem, and totals the matched distances.  The "most stable"
        run is the one whose total distance to every other run is
        minimal; the returned vector gives every run's distance to that
        reference.

        Args:
            centroids_list: List of ``(k, n_features)`` arrays — one per
                independent clustering run within the same group.

        Returns:
            1-D array of length ``len(centroids_list)``: distance from
            each run to the most-stable reference run.
        """
        num_runs = len(centroids_list)
        distances = np.zeros((num_runs, num_runs))

        for i in range(num_runs):
            for j in range(i + 1, num_runs):
                cost_matrix = np.linalg.norm(
                    centroids_list[i][:, np.newaxis, :] - centroids_list[j],
                    axis=2,
                )
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                total_cost = cost_matrix[row_ind, col_ind].sum()
                distances[i, j] = total_cost
                distances[j, i] = total_cost

        row_totals = distances.sum(axis=0)
        most_stable_idx = int(np.argmin(row_totals))
        return distances[:, most_stable_idx]

    def calculate_and_assign_instability(self) -> None:
        """Compute and assign per-run ``instability`` values back to ``self.df``."""
        if self.df is None:
            raise RuntimeError("Call analyze() or load_df() before computing instability.")

        groups = self.df[["k_number", "reduction_percent"]].drop_duplicates()

        for _, row in groups.iterrows():
            k_number = row["k_number"]
            reduction_percent = row["reduction_percent"]
            centroids_list = self.load_centroids_for_analysis(reduction_percent, k_number)
            instability_values = self.calculate_instability(centroids_list)

            condition = (
                (self.df["k_number"] == k_number)
                & (self.df["reduction_percent"] == reduction_percent)
            )
            for i, (index, _) in enumerate(self.df[condition].iterrows()):
                self.df.at[index, "instability"] = instability_values[i]

    # ─────────────────────────────────────────────────────────────────
    #  Orchestration and DataFrame I/O
    # ─────────────────────────────────────────────────────────────────

    def analyze(self) -> None:
        """Load all metadata and populate the instability column."""
        self.df = self.load_data()
        self.calculate_and_assign_instability()

    def save_df(self, save_path: str | Path) -> None:
        """Write ``self.df`` to CSV."""
        if self.df is None:
            raise RuntimeError("No DataFrame loaded.")
        self.df.to_csv(save_path, index=False)
        print(f"DataFrame saved to {save_path}")

    def load_df(self, load_path: str | Path) -> pd.DataFrame:
        """Read ``self.df`` from CSV and return it."""
        self.df = pd.read_csv(load_path)
        print(f"DataFrame loaded from {load_path}")
        return self.df

    # ─────────────────────────────────────────────────────────────────
    #  Most-stable centroid lookup and post-processing
    # ─────────────────────────────────────────────────────────────────

    def find_most_stable_centroids(
        self, k_number: int, reduction_percent: float,
    ) -> tuple[np.ndarray, str]:
        """Return the centroid array of the minimum-instability run in a group.

        On a tie, the first matching row is returned.
        """
        if self.df is None:
            raise RuntimeError("Call analyze() or load_df() first.")

        filtered = self.df[
            (self.df["k_number"] == k_number)
            & (self.df["reduction_percent"] == reduction_percent)
        ]
        most_stable = filtered.loc[filtered["instability"].idxmin()]

        with open(most_stable["file_path"], "r") as f:
            content = json.load(f)
        centroids = np.array(content["centroids"])

        return centroids, most_stable["file_path"]

    @staticmethod
    def de_zscore_centroids(
        centroids: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
    ) -> np.ndarray:
        """Reverse z-scoring of ``centroids`` using per-feature ``mu`` and ``sigma``."""
        out = np.zeros_like(centroids)
        for i in range(centroids.shape[1]):
            out[:, i] = centroids[:, i] * sigma[i] + mu[i]
        return out

    @staticmethod
    def normalize_centroids(centroids: np.ndarray) -> np.ndarray:
        """Normalise each feature to ``[-1, 1]`` by dividing by its absolute max."""
        abs_max = np.max(np.abs(centroids), axis=0)
        return centroids / abs_max


class ClusterPlotter:
    """Plot quality / instability panels and centroid radar charts."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

    def plot_metric(self, y_metric: str, log_scale: bool = False) -> plt.Figure:
        """Line plot of ``y_metric`` against ``k_number`` per ``reduction_percent``.

        Args:
            y_metric:  Column name to plot on the y-axis.
            log_scale: Whether to log-scale the y-axis.
        """
        sns.set_theme(context="talk", style="whitegrid", palette="colorblind")

        if (
            y_metric == "analysis_duration"
            and self.dataframe[y_metric].dtype == "object"
        ):
            self.dataframe[y_metric] = pd.to_timedelta(self.dataframe[y_metric])
            self.dataframe[y_metric] = self.dataframe[y_metric].dt.total_seconds() / 60

        fig = plt.figure(figsize=(10, 6))
        markers = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "+", "x"]
        n_reductions = len(self.dataframe["reduction_percent"].unique())

        ax = sns.lineplot(
            data=self.dataframe,
            x="k_number", y=y_metric, hue="reduction_percent",
            errorbar=("ci", 95), estimator="median",
            style="reduction_percent",
            markers=markers[:n_reductions],
            dashes=False, linewidth=2.5,
        )

        if log_scale:
            ax.set_yscale("log")

        ax.grid(which="major", linestyle="-", linewidth=0.5, color="gray")
        ax.grid(which="minor", linestyle=":", linewidth=0.5, color="lightgray")

        legend = ax.legend(
            title="Reduction Percent",
            loc="upper right",
            bbox_to_anchor=(1.15, 1),
            borderaxespad=0.0,
        )
        frame = legend.get_frame()
        frame.set_color("white")
        frame.set_edgecolor("gray")

        plt.title(f"{y_metric.capitalize()} by K-Number with Different Reduction Percentages")
        plt.xlabel("K-Number")
        plt.ylabel(y_metric.capitalize().replace("_", " "))
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_radar_charts(
        centroids: np.ndarray,
        feature_labels: list[str],
        normalise: bool = True,
    ) -> plt.Figure:
        """Plot one polar radar chart per cluster on a shared grid.

        Args:
            centroids:      ``(n_clusters, n_features)`` array.
            feature_labels: Axis labels (one per feature).
            normalise:      If True, set y-limits to ``[-1, 1]`` and draw
                            a thicker zero ring.
        """
        num_vars = len(feature_labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        loop_centroids = np.concatenate(
            (centroids, centroids[:, [0]]), axis=1,
        )  # close the polygon
        loop_angles = angles + angles[:1]

        n_clusters = centroids.shape[0]
        ncols = 2
        nrows = (n_clusters + ncols - 1) // ncols

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(ncols * 6, nrows * 6),
            subplot_kw=dict(polar=True),
        )
        axs = np.atleast_1d(axs).flatten()

        for i in range(n_clusters):
            ax = axs[i]
            ax.plot(loop_angles, loop_centroids[i], linewidth=1, linestyle="solid",
                    label=f"Cluster {i + 1}")
            ax.fill(loop_angles, loop_centroids[i], alpha=0.1)

            ax.set_xticks(angles)
            ax.set_xticklabels(feature_labels, color="grey", size=12)

            if normalise:
                ax.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax.set_ylim(-1, 1)
                ax.plot(loop_angles, [0] * len(loop_angles),
                        "k-", linewidth=2)
                ax.set_title(f"Cluster {i + 1}", size=16, color="blue", y=1.1)

        for i in range(n_clusters, len(axs)):
            axs[i].axis("off")

        plt.tight_layout()
        return fig
