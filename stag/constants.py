# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — constants                                                ║
# ║  « one source of truth for palettes, paths, and figure rules »   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Central configuration for all STAG analyses and figures.       ║
# ║  Import this module instead of hardcoding hex values, paths,    ║
# ║  feature names, or behavioural-category labels.                 ║
# ║                                                                  ║
# ║  Wong (2011) colourblind-safe palette with semantic mappings    ║
# ║  to the eight prototypical movements (PM0–PM7) identified in    ║
# ║  the k = 8 representative clustering run.                       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Central configuration for STAG analyses and figures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from stag.local_paths import get_path_obj as _local

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from pandas import DataFrame


# ┌────────────────────────────────────────────────────────────┐
# │ Sensor / sampling constants  « hardware-fixed values »     │
# └────────────────────────────────────────────────────────────┘

FPS: int = 50
"""Accelerometer sampling rate in Hz."""

GPS_FPS: float = 0.5
"""GPS sampling rate in Hz (0.5 Hz = one fix every 2 s)."""

VIDEO_FPS: int = 30
"""Ground-truth video sampling rate in Hz."""

ACCELEROMETER_RANGE_G: int = 16
"""Accelerometer dynamic range in units of Earth gravity (±16 g)."""

FEATURE_LABELS: tuple[str, ...] = (
    "Head_X", "Head_Y", "Head_Z",
    "Ear_X", "Ear_Y", "Ear_Z",
)
"""Six accelerometer feature names in the order used by the clustering input matrix."""


# ┌────────────────────────────────────────────────────────────┐
# │ Wong (2011) palette  « colourblind-safe base colours »     │
# └────────────────────────────────────────────────────────────┘

WONG: dict[str, str] = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermilion":      "#D55E00",
    "reddish_purple": "#CC79A7",
}
"""Wong (2011, Nature Methods 8: 441) colourblind-safe palette."""


# ┌────────────────────────────────────────────────────────────┐
# │ Prototypical movements  « k = 8 representative run »       │
# └────────────────────────────────────────────────────────────┘

PM_NAMES: dict[int, str] = {
    0: "quiescent",
    1: "resting",
    2: "ear_flick_out",
    3: "resting_active",
    4: "ear_flick_back",
    5: "ear_flick_composite",
    6: "grazing_stepping",
    7: "stationary_grazing",
}
"""Canonical PM index → snake_case label mapping."""

PM_DISPLAY_NAMES: dict[int, str] = {
    0: "Quiescent",
    1: "Resting",
    2: "Ear flick (out)",
    3: "Resting, active",
    4: "Ear flick (back)",
    5: "Ear flick (composite)",
    6: "Stepping, grazing",
    7: "Stationary, grazing",
}
"""Long display name per PM — keeps the ear-flick subtype distinction.

Use this for axes where each cluster is shown on its own row/column
(centroid heatmaps, confusion matrices, transition matrices).
"""

PM_DISPLAY_NAMES_SHORT: dict[int, str] = {
    0: "Quiescent",
    1: "Resting",
    2: "Ear flick",
    3: "Resting, active",
    4: "Ear flick",
    5: "Ear flick",
    6: "Stepping, grazing",
    7: "Stationary, grazing",
}
"""Short display name per PM — collapses the three ear-flick subtypes
to a single "Ear flick" label.

Use this for legends and figures where the three ear-flick prototypes
share a colour (see :data:`PM_COLOURS`) and read as one behaviour.
"""

PM_CATEGORY: dict[int, str] = {
    0: "inactive",
    1: "inactive",
    3: "inactive",
    6: "grazing",
    7: "grazing",
    2: "ear_flick",
    4: "ear_flick",
    5: "ear_flick",
}
"""PM index → behavioural-category label (three families)."""

PM_COLOURS: dict[int, str] = {
    0: "#66ddcc",  # Quiescent
    1: "#1d8475",  # Resting
    2: "#e0ce61",  # Ear flick (shared)
    3: "#96d3ed",  # Resting, active
    4: "#e0ce61",  # Ear flick (shared)
    5: "#e0ce61",  # Ear flick (shared)
    6: "#86771a",  # Stepping, grazing
    7: "#8497b0",  # Stationary, grazing
}
"""Per-PM colour assignment used in every manuscript figure.

PM2, PM4, and PM5 share ``#e0ce61`` so the three ear-flick subtypes
read as one visual category.  These colours are the authoritative
figure palette and supersede any Wong-derived category palette.
"""

PM_CATEGORY_COLOURS: dict[str, str] = {
    "inactive":  "#3F7FB5",   # blue
    "grazing":   "#4E8C3A",   # green
    "ear_flick": "#D96828",   # orange
}
"""Per-behavioural-family colour swatch — lab canonical palette.

These three colours are the agreed-upon family swatches for every
figure that groups the eight PMs into their behavioural families
(Inactive = PM 0 + PM 1 + PM 3, Grazing = PM 6 + PM 7, Ear flick =
PM 2 + PM 4 + PM 5).  They are deliberately *not* derived from
:data:`PM_COLOURS` (which uses the Wong palette for per-PM plotting):
the family-level chart asks a different question (three families,
not eight prototypes) and the lab uses a distinct palette to make
that distinction visually obvious in posters, slides, and the
manuscript figures.

Use :data:`PM_COLOURS` for per-PM plotting; use this dict for
category-level legends and category-grouped bar charts where one
swatch per family is wanted.
"""


# ┌────────────────────────────────────────────────────────────┐
# │ Figure defaults  « SVG + PNG + CSV triple output »         │
# └────────────────────────────────────────────────────────────┘

FIGURE_DPI: int = 200
"""Raster export resolution (PNG)."""

FIGURE_SIZE_SINGLE: tuple[float, float] = (3.5, 2.8)
"""Single-column Elsevier figure size in inches."""

FIGURE_SIZE_DOUBLE: tuple[float, float] = (7.2, 4.5)
"""Two-column (full-width) Elsevier figure size in inches."""

FONT_FAMILY: str = "DejaVu Sans"
"""Default sans-serif font for figures."""

# Matplotlib rcParams applied by :func:`apply_figure_defaults`.
_FIGURE_RC: dict[str, object] = {
    "svg.fonttype":      "none",      # editable text in Inkscape
    "savefig.dpi":       FIGURE_DPI,
    "savefig.bbox":      "tight",
    "font.family":       FONT_FAMILY,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
}


def apply_figure_defaults() -> None:
    """Apply STAG figure rcParams to the current matplotlib session."""
    plt.rcParams.update(_FIGURE_RC)


# ┌────────────────────────────────────────────────────────────┐
# │ Path templates  « default output layout »                  │
# └────────────────────────────────────────────────────────────┘

RESULTS_DIR_DEFAULT: Path = Path("results")
"""Default top-level output directory (relative to working dir)."""

FIGURES_SUBDIR: str = "figures"
TABLES_SUBDIR: str = "tables"


# ┌────────────────────────────────────────────────────────────┐
# │ Canonical data paths  « DINZ deer-2024 archive »           │
# └────────────────────────────────────────────────────────────┘
#
# See [[Data Files — Source of Truth]] in the DINZ Obsidian folder
# for the full file map (what is canonical, derived, legacy, or
# corrupt).  The paths below are the ones analysis scripts default
# to; pass --meta-dir / --data-file on the CLI to override.

HCS_SOURCE_DIR: Path = _local(
    "hcs_source",
    default="<hcs_source not configured - see local_paths.template.json>",
)
"""Read-only network archive of the 2024 deer dataset (3.0 TB).
Too slow for direct analysis — mirror the curated subset locally.
Resolved by :mod:`stag.local_paths`: ``STAG_HCS_DIR`` env var,
then ``local_paths.json`` ``hcs_source`` field, then the placeholder
default (which will crash any downstream read with a clear path)."""

LOCAL_DATA_DIR: Path = _local(
    "data_root",
    default="<data_root not configured - see local_paths.template.json>",
)
"""Local working copy on the NVMe data drive.  Tier-1 footprint
≈ 43 GB.  All path constants below are anchored here.  Resolved by
:mod:`stag.local_paths`: ``STAG_DATA_DIR`` env var, then
``local_paths.json`` ``data_root`` field, then placeholder default."""

RAW_CLUSTERING_INPUT: Path = LOCAL_DATA_DIR / "clust_data_raw_20240412.npy"
"""Raw 8-column input the SLURM clustering actually read.
Shape ``(204_554_618, 8)`` float64.  Columns 0–5 are the six
accelerometer axes; columns 6–7 are GPS-derived speed and
tortuosity (excluded from clustering)."""

MAXABS_CLUSTERING_INPUT: Path = LOCAL_DATA_DIR / "clust_data_maxabs_6col.npy"
"""Six-column MaxAbs-scaled feature matrix derived from
:data:`RAW_CLUSTERING_INPUT` by
``scripts/preprocess_clustering_data.py``.  Shape
``(204_554_618, 6)`` float64.  Each column is divided by its
absolute maximum, mapping the data to [-1, 1] per column —
reproducing the 2024 SLURM pipeline's normalisation exactly
(MaxAbsScaler + col-5 ±7.99 clip).  This is the file every
internal- and external-validation analysis consumes."""

MAXABS_SCALER_CSV: Path = LOCAL_DATA_DIR / "clust_data_maxabs_6col.maxabs.csv"
"""Per-column max-abs divisors used to produce
:data:`MAXABS_CLUSTERING_INPUT`.  Written alongside the .npy
by the preprocess script — needed to invert the scaling back
to physical units for centroid interpretation."""

DEER_DB: Path = LOCAL_DATA_DIR / "deer_data_gps.db"
"""SQLite copy of the canonical deer-2024 DB (~58 GB) — six tables:
``accelerometer_data`` (~222 M rows), ``cluster_labels`` (~204.5 M rows,
FK ``acc_id`` → ``accelerometer_data.data_id``), ``trajectory_data``
(GPS, ~207 M rows after upsampling), ``deer_info`` (26 animals),
``video_observation_reference`` (926 clips), ``video_availability``
(2.8 M rows).  All required composite/FK indexes are already in
place.  See ``scripts/cache_label_timeline.py`` for the canonical
join that aligns DB rows with the saved ``labels.npy``."""

LABEL_TIMELINE_DEER_IDS: Path = LOCAL_DATA_DIR / "label_timeline_deer_ids.npy"
"""Per-sample ``deer_id`` aligned with the saved k=8 ``labels.npy``
(shape ``(204_554_618,)`` int8).  Built once by
``scripts/cache_label_timeline.py`` from the DEER_DB join
``cluster_labels.acc_id`` → ``accelerometer_data.data_id``."""

LABEL_TIMELINE_TIMESTAMPS: Path = LOCAL_DATA_DIR / "label_timeline_timestamps.npy"
"""Per-sample wall-clock timestamp aligned with the saved k=8
``labels.npy`` (shape ``(204_554_618,)`` int64 nanoseconds since
the Unix epoch, NZ local time as stored in the DB).  Built once
by ``scripts/cache_label_timeline.py``."""

CLUSTER_RESULTS_DIR: Path = LOCAL_DATA_DIR / "cluster_results" / "deer6raw"
"""Root of the per-fit metadata + centroids + labels tree
produced by the Aoraki SLURM sweep.  Per-fit JSONs and centroid
arrays are present in full; labels are present for the 24
representative runs only."""

CANONICAL_K8_LABELS: Path = (
    CLUSTER_RESULTS_DIR / "delSize_0" / "k_8" / "labels"
    / "deer6raw_labels_k8_delSize0_partA.npy"
)
"""Manuscript-aligned k=8 labels.npy (Partition A).

Of the 50 fits saved at ``delSize_0/k_8``, 17 converged to the
basin whose centroids match :data:`centroid_label_info` exactly and
whose cluster IDs are in manuscript PM order
(0 = Quiescent, 1 = Resting, … 7 = Stationary grazing).  None of
those 17 had their labels mirrored from the original sweep, so
this file is regenerated locally by nearest-manuscript-centroid
assignment on :data:`MAXABS_CLUSTERING_INPUT` — bit-equivalent to
the converged k-means assignment for that basin, prevalences match
the manuscript table to within 0.013 % per PM.

Use this constant — not a glob of the labels directory — wherever
downstream code wants "the k=8 labels."""


def save_figure(
    fig: "Figure",
    stem: str,
    output_dir: Path,
    data: "DataFrame | None" = None,
) -> None:
    """Export a figure as SVG + PNG with an optional CSV companion table.

    Args:
        fig:        Matplotlib figure to save.
        stem:       Filename stem (no extension).
        output_dir: Target directory (created if needed).
        data:       Optional dataframe; written as ``<stem>.csv`` alongside.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_figure_defaults()

    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=FIGURE_DPI)

    if data is not None:
        csv_path = output_dir / f"{stem}.csv"
        data.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
