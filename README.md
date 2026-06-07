# STAG — Sensor-based Tracking and Analysis of Gait

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/zerotonin/stag/actions/workflows/tests.yml/badge.svg)](https://github.com/zerotonin/stag/actions/workflows/tests.yml)
[![Documentation](https://github.com/zerotonin/stag/actions/workflows/docs.yml/badge.svg)](https://zerotonin.github.io/stag/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19106434.svg)](https://doi.org/10.5281/zenodo.19106434)

An unsupervised machine-learning pipeline for classifying farmed red deer
(*Cervus elaphus*) behaviour from wearable tri-axial accelerometer data.

STAG discovers prototypical movement patterns directly from sensor streams
using *k*-means clustering, chains them into higher-order behavioural
sequences via a first-order Markov transition model, and runs on a 16 MHz
microcontroller at over 4 × 10⁸ classifications per second — no GPU,
cloud link, or labelled training data required at inference time.

---

## Pipeline overview

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `stag.sync` | Synchronise head & ear accelerometer streams via calibration-drop events |
| 2 | `stag.database` | Ingest synchronised `.h5` files into a SQLite database (SQLAlchemy ORM) |
| 3 | `stag.gps` | Compute ground speed and path tortuosity from GPS fixes (NZMG projection) |
| 4 | `stag.clustering` | GPU-accelerated *k*-means with contiguous leave-out stability analysis |
| 5 | `stag.analysis` | Transition matrices, bout statistics, and Markov super-prototypes |

## Repository structure

```
stag/
├── stag/                   # Python package
│   ├── sync/               # Sensor synchronisation
│   ├── database/           # SQLAlchemy ORM & database construction
│   ├── gps/                # GPS trajectory analysis & plotting
│   ├── clustering/         # k-means clustering & meta-analysis
│   ├── analysis/           # Label analysis, Markov transitions, preprocessing
│   └── utils/              # Logging, filename generation, helpers
├── scripts/                # Runnable entry-point scripts
├── slurm/                  # HPC job submission scripts (NeSI / Aoraki)
├── data/                   # Deer code CSVs and auxiliary data
│   └── deer_codes/         # Animal identification lookup tables
├── docs/                   # Sphinx documentation source
├── tests/                  # Unit tests (to be expanded)
├── CITATION.cff            # Machine-readable citation metadata
├── LICENSE                 # MIT License
├── pyproject.toml          # Build system & dependencies
└── environment.yml         # Conda environment specification
```

## Installation

### With conda (recommended for GPU clustering)

```bash
conda env create -f environment.yml
conda activate stag
pip install -e .
```

### With pip (CPU-only, no RAPIDS)

```bash
pip install -e .
```

For GPU-accelerated clustering you additionally need
[RAPIDS cuML](https://rapids.ai/) installed in your environment.

## Quick start

The pipeline runs in five stages. Each stage is a single command from the
repo root and produces the input file the next stage consumes; paths are
resolved from `stag/constants.py` and can be overridden via the CLI flags
shown for each stage. The full sequence reproduces the manuscript's
Sprint 1–3 analyses end-to-end.

### 0. Install + smoke test

```bash
conda env create -f environment.yml && conda activate stag
pip install -e ".[dev]"
pytest                                                # 80+ unit tests
```

### 1. Synchronise ear + head IMU signals into a SQLite database

```python
from stag.sync.data_sync import BetterDataSync

syncer = BetterDataSync(
    deer_id="R1_D1",
    head_data=head_df,
    ear_data=ear_df,
    window_dict={"start": 0, "end": 50000},
)
syncer.run_synchronization()                          # writes deer_data_gps.db
```

The synchroniser locates three calibration drops in both signals,
solves for the time offset, fuses ear + head into a single
6-dimensional accelerometer stream, and writes one row per sample into
the SQLite database at `LOCAL_DATA_DIR/deer_data_gps.db`. One deer per
call; loop over the deer-code lookup CSV in `data/deer_codes/` to fill
the full cohort.

### 2. Pre-process the database into a clustering-ready array

```bash
python scripts/preprocess_clustering_data.py          # → clust_data_maxabs_6col.npy
```

Streams the synchronised database, clips column 5 to ± 7.99 g (the per-
animal sensor saturation), applies the per-column MaxAbs scaler whose
divisors are recorded in `clust_data_maxabs_6col.maxabs.csv`, and emits
a memmap-friendly `.npy` that the GPU clustering reads directly.

### 3. Cluster and validate (internal metrics + Figure 2)

```bash
python scripts/run_internal_metrics.py --chosen-k 8   # → results/internal_metrics/
```

Sweeps k = 2 .. 30 under the contiguous-leave-out protocol (50 cut
positions per k), computes Calinski–Harabasz, Hungarian-matched
centroid stability, stratified Silhouette, and Kneedle elbow on the
inertia curve, and renders the four-panel Figure 2 plus a one-row
`selection_summary.csv` recording the k = 8 choice and its bounds.

### 4. External validation against video annotations

```bash
python scripts/analyse_groundtruthing_register.py <register.csv>
```

Parses the observer's free-text `Behaviour` column from a DINZ
groundtruthing register into a canonical 15-token behavioural
vocabulary, computes per-prototype agreement with Wilson 95 % CIs, and
renders the cluster × token confusion matrix and the PM × PM external
confusion matrix under a conditional decision rule. Writes one SVG +
PNG + CSV trio per figure. The token-overlap framing is robust to
cluster-ID drift between historical k-means fits, which is why this
script — not a Hungarian-aligned ARI / NMI pipeline — is the
canonical external-validation tool.

### 5. Sequence statistics (super-prototypes, circadian, ear-flick)

```bash
python scripts/cache_label_timeline.py                # one-off helper
python scripts/run_sequence_stats.py --n-shuffles 1000 --percentile 99.9
                                                      # → results/sprint3/
```

Run-length-encodes the per-sample cluster labels into bout streams,
generates the first-order Markov shuffle null distribution, identifies
super-prototypes that beat the joint 99.9th-percentile AND
Benjamini–Hochberg q < 0.05 threshold, and writes
`super_prototype_triplets.csv` plus the day/night Wilcoxon table and
per-animal hourly time budgets.

### Optional: GPS trajectory + tortuosity

```bash
python scripts/run_tortuosity.py R4_D1 bart_paths     # one deer, one path-system
```

Loads the GPS track for the named deer, computes Hausdorff-corrected
tortuosity and speed, and saves the trajectory + summary plots.

Each stage is independent — re-run any stage after changing its
parameters without invalidating the earlier ones, as long as the input
file from the previous stage is still on disk.

For the SLURM-orchestrated versions of stages 2 and 3 (cohort-scale
runs on the Aoraki HPC cluster), see [`slurm/`](slurm/).

## Hardware deployment

The trained nearest-centroid classifier (`stag.embedded.nearest_centroid`,
Q4.12 fixed-point, K = 8, D = 6) was benchmarked on ten microcontrollers
spanning every architectural class currently used in animal-borne
biologger and maker-board deployments. The identical C source was
compiled with the vendor-recommended `gcc` cross-toolchain at `-Os -g`
and run under a cycle-tracking emulator (simavr for AVR, Renode for
Cortex-M, mspdebug for MSP430, Espressif QEMU for Xtensa). Per-MCU
firmware, linker scripts, and Python runners live under
[`stag/embedded/benchmark/`](stag/embedded/benchmark/).

| MCU | Architecture | Clock | Cyc / call | Throughput | × 50 Hz |
|---|---|---:|---:|---:|---:|
| ATmega328P (Arduino Uno)             | 8-bit AVR        |  16 MHz |  3 554    | 4.5 k / s   |     90× |
| ATmega32U4 (Pro Micro / Feather)     | 8-bit AVR        |  16 MHz |  3 554    | 4.5 k / s   |     90× |
| ATmega2560 (Arduino Mega)            | 8-bit AVR        |  16 MHz |  3 730    | 4.3 k / s   |     86× |
| MSP430G2553 (TI LaunchPad)           | 16-bit MSP430    |  16 MHz | 31 652    | 506 / s     |     10× |
| SAMD21G18A (Feather M0 / MKR Zero)   | Cortex-M0+       |  48 MHz |    798 ‡  |  60 k / s   |  1 200× |
| nRF52840 (Feather nRF52840)          | Cortex-M4F       |  64 MHz |    415 ‡  | 154 k / s   |  3 080× |
| RP2040 (Raspberry Pi Pico)           | Cortex-M0+       | 133 MHz |    798 ‡  | 167 k / s   |  3 340× |
| STM32F407 (STM32F4-Discovery)        | Cortex-M4F       | 168 MHz |    415 ‡  | 405 k / s   |  8 100× |
| ESP32 (Espressif WROOM-32)           | Xtensa LX6       | 240 MHz |    188 §  | 1.28 M / s  | 25 600× |
| i.MX RT1064 (NXP RT106x / Teensy)    | Cortex-M7        | 600 MHz |    415 ‡† | 1.45 M / s  | 29 000× |

Every silicon class clears the 50 Hz inertial-sampling budget by at
least an order of magnitude, with the value-line MSP430G2553 (no
hardware multiplier; software-emulated `__mulhi3`) being the slowest
at a 10× margin.

<sub>‡ Renode `cpu ExecutedInstructions`, single-issue — silicon will be the same or faster. † Cortex-M7's dual-issue pipeline typically realises ≈ 1.3 IPC on integer workloads; RT106x silicon throughput exceeds the reported value by an estimated 25 %. § Espressif QEMU virtual CCOUNT under `-icount shift=auto`; pipeline and cache effects not modelled. AVR rows are simavr-measured hardware cycles.</sub>

## Citation

If you use STAG in your research, please cite the software and the
accompanying journal article:

```bibtex
@misc{matthews2026stag_software,
  title        = {STAG - Sensor-based Tracking and Analysis of Gait},
  author       = {Matthews, Alexander R. H. and Matthews, Lindsay R. and
                  Geurten, Bart R. H.},
  year         = {2026},
  doi          = {10.5281/zenodo.19106434},
  url          = {https://doi.org/10.5281/zenodo.19106434},
  note         = {Concept DOI - resolves to the latest version on Zenodo}
}

@article{matthews2026stag,
  title   = {Behavioural Phenotyping in Red Deer: Machine Learning
             Classification of Accelerometer Data from Micro-Movements
             to Grazing},
  author  = {Matthews, Alexander R. H. and Matthews, Lindsay R. and
             Geurten, Bart R. H.},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026}
}
```

See also [`CITATION.cff`](CITATION.cff) for machine-readable citation
metadata.  Each tagged release is archived on Zenodo with its own
version-specific DOI; cite the version DOI for reproducibility, or the
concept DOI above to always resolve to the latest release.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for
details.

## Authors

- **Alexander R. H. Matthews** — Department of Zoology, University of
  Otago, Dunedin, New Zealand
- **Lindsay R. Matthews** — Matthews Research International LP, New Zealand
- **Bart R. H. Geurten** — Department of Zoology, University of Otago,
  Dunedin, New Zealand *(corresponding author: bart.geurten@otago.ac.nz)*

## Development history

This pipeline was developed at the Department of Zoology, University of
Otago. Early development — including the original per-replicate
exploratory Jupyter notebooks, the per-deer data-merging notebooks, and
the tortuosity-analysis notebooks — took place under
[github.com/alexrhmatthews/headshake_project](https://github.com/alexrhmatthews/headshake_project)
and remains available there for provenance. The codebase was then
reorganised and renamed to STAG for publication: every function the
original notebooks performed has been refactored into the production
Python package (`stag.sync.data_sync` and `stag.sync.utils` for the
ear/head IMU peak-matching and 3-drop calibration sync; `stag.gps.*`
for the GPS and tortuosity analyses; `stag.database.*` for the
SQLAlchemy ORM and SQLite consolidation; `stag.clustering.*` and
`stag.analysis.*` for k-means and the first-order Markov sequence
analysis) and is exercised by the `scripts/` entry points and the
`tests/` suite. The notebooks themselves are therefore not duplicated
in this repository.
