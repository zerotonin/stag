"""
Behavioural sequence analysis from cluster label time series.

This module implements the :class:`LabelAnalyser`, which takes the
per-time-step cluster assignments produced by the k-means stage and
computes:

- Percentage prevalence of each prototypical movement.
- Bout durations (mean ± SEM) for contiguous runs of the same label.
- A first-order transition matrix (the basis for HMM super-prototypes).
- Short-sequence filtering to merge spurious single-frame labels into
  adjacent bouts (after Braun & Geurten, 2010).

Results are saved as a single JSON file for downstream plotting and
circadian analysis.
"""

import json

import numpy as np
from tqdm import tqdm


class LabelAnalyser:
    """Analyse cluster labels for behavioural statistics and transitions.

    Parameters
    ----------
    file_path : str
        Path to a ``.npy`` file containing the integer label array.
    fps : int, optional
        Sampling rate in Hz, used to convert bout lengths to seconds.
        Default ``50``.

    Attributes
    ----------
    IDX : numpy.ndarray
        The label array (modified in place by :meth:`filterIDX`).
    fps : int
        Frames per second.
    cen_num : int
        Number of unique cluster labels.
    label_num : int
        Total number of time steps.
    """

    def __init__(self, file_path, fps=50):
        self.IDX = np.load(file_path)
        self.fps = fps
        self.cen_num = self.IDX.max() + 1
        self.label_num = self.IDX.shape[0]

    def filterIDX(self, cutoff):
        """Merge short label runs into neighbouring bouts.

        Sequences shorter than *cutoff* frames are absorbed by the
        adjacent bout (previous or next) that is longer, following the
        approach of Braun & Geurten (2010).

        Parameters
        ----------
        cutoff : int
            Minimum bout length (in frames) to retain.
        """
        IDX_diff = np.diff(self.IDX, prepend=self.IDX[0], append=self.IDX[-1])
        IDX_changes = np.where(IDX_diff != 0)[0]
        IDX_starts = np.r_[0, IDX_changes + 1]
        IDX_ends = np.r_[IDX_changes, len(self.IDX) - 1]
        IDX_durations = IDX_ends - IDX_starts + 1

        for i, duration in tqdm(enumerate(IDX_durations), desc="filtering IDX"):
            if duration <= cutoff:
                prev_len = IDX_durations[i - 1] if i > 0 else np.inf
                next_len = IDX_durations[i + 1] if i < len(IDX_durations) - 1 else np.inf
                if i == 0 or prev_len > next_len:
                    if i < len(IDX_durations) - 1:
                        self.IDX[IDX_starts[i] : IDX_ends[i] + 1] = self.IDX[IDX_starts[i + 1]]
                else:
                    self.IDX[IDX_starts[i] : IDX_ends[i] + 1] = self.IDX[IDX_starts[i - 1]]

    def get_percentage(self):
        """Compute the prevalence of each label as a percentage.

        Returns
        -------
        numpy.ndarray
            Array of length ``cen_num`` with percentage values.
        """
        occurrences = np.bincount(self.IDX, minlength=self.cen_num)
        return (occurrences / self.label_num) * 100

    def _get_train_lengths(self, ignore_ones=True):
        """Compute the length of each contiguous label run.

        Parameters
        ----------
        ignore_ones : bool, optional
            If ``True``, single-frame runs are excluded.
            Default ``True``.

        Returns
        -------
        list of list of int
            One sub-list per label, containing bout lengths in frames.
        """
        current_number = self.IDX[0]
        current_length = 1
        trains = [[] for _ in range(self.cen_num)]

        for num in tqdm(self.IDX[1:], desc="bout durations"):
            if num == current_number:
                current_length += 1
            else:
                if current_length > 1 or not ignore_ones:
                    trains[current_number].append(current_length)
                current_number = num
                current_length = 1
        if current_length > 1 or not ignore_ones:
            trains[current_number].append(current_length)
        return trains

    def get_mean_durations(self):
        """Compute mean bout duration and SEM for each label.

        Returns
        -------
        list of tuple of (float, float)
            Each tuple is ``(mean_seconds, sem_seconds)``.
        """
        means = []
        sems = []
        trains = self._get_train_lengths(ignore_ones=False)
        for train in trains:
            arr = np.array(train)
            means.append(np.mean(arr) / self.fps)
            std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
            sems.append((std / np.sqrt(len(arr))) / self.fps)
        return list(zip(means, sems))

    def get_transitions(self):
        """Build the first-order transition matrix.

        Returns
        -------
        numpy.ndarray
            Square matrix of shape ``(cen_num, cen_num)`` where entry
            ``(i, j)`` counts transitions from label *i* to label *j*.
        """
        transitions = np.zeros((self.cen_num, self.cen_num))
        for i in tqdm(range(len(self.IDX) - 1), desc="transitions"):
            transitions[self.IDX[i], self.IDX[i + 1]] += 1
        return transitions

    def save_results_to_json(self, file_path, durations, percentages, transitions):
        """Write analysis results to a JSON file.

        Parameters
        ----------
        file_path : str
            Output JSON path.
        durations : list of tuple
            From :meth:`get_mean_durations`.
        percentages : numpy.ndarray
            From :meth:`get_percentage`.
        transitions : numpy.ndarray
            From :meth:`get_transitions`.
        """
        data = {
            "centroids": [],
            "transition_matrix": transitions.tolist(),
        }
        for i, ((mean, sem), pct) in enumerate(zip(durations, percentages), start=1):
            data["centroids"].append({
                "centroid": i,
                "percentage": float(pct),
                "duration_sec_mean": mean,
                "duration_sec_sem": sem,
            })
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Results saved to {file_path}")

    def main(self, cutoff, save_path):
        """Run the full label analysis pipeline.

        Parameters
        ----------
        cutoff : int
            Minimum bout length for filtering (in frames).
        save_path : str
            Path for the output JSON file.
        """
        self.filterIDX(cutoff)
        durations = self.get_mean_durations()
        percentages = self.get_percentage()
        transitions = self.get_transitions()
        self.save_results_to_json(save_path, durations, percentages, transitions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STAG label sequence analysis.")
    parser.add_argument("label_file", help="Path to .npy label array.")
    parser.add_argument("output", help="Path for output JSON.")
    parser.add_argument("--fps", type=int, default=50, help="Sampling rate (Hz).")
    parser.add_argument("--cutoff", type=int, default=2, help="Min bout length (frames).")
    args = parser.parse_args()

    analyser = LabelAnalyser(args.label_file, fps=args.fps)
    analyser.main(args.cutoff, args.output)
