# STAG — Sensor-based Tracking and Analysis of Gait

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An unsupervised machine-learning pipeline for classifying farmed red deer
(*Cervus elaphus*) behaviour from wearable tri-axial accelerometer data.

STAG discovers prototypical movement patterns directly from sensor streams
using *k*-means clustering, chains them into higher-order behavioural
sequences via a Hidden Markov Model, and runs on a 16 MHz microcontroller
at over 4 × 10⁸ classifications per second — no GPU, cloud link, or
labelled training data required at inference time.

---

## Pipeline overview

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `stag.sync` | Synchronise head & ear accelerometer streams via calibration-drop events |
| 2 | `stag.database` | Ingest synchronised `.h5` files into a SQLite database (SQLAlchemy ORM) |
| 3 | `stag.gps` | Compute ground speed and path tortuosity from GPS fixes (NZMG projection) |
| 4 | `stag.clustering` | GPU-accelerated *k*-means with contiguous leave-out stability analysis |
| 5 | `stag.analysis` | Transition matrices, bout statistics, and HMM super-prototypes |

## Repository structure

```
stag/
├── stag/                   # Python package
│   ├── sync/               # Sensor synchronisation
│   ├── database/           # SQLAlchemy ORM & database construction
│   ├── gps/                # GPS trajectory analysis & plotting
│   ├── clustering/         # k-means clustering & meta-analysis
│   ├── analysis/           # Label analysis, HMM, preprocessing
│   └── utils/              # Logging, filename generation, helpers
├── scripts/                # Runnable entry-point scripts
├── slurm/                  # HPC job submission scripts (NeSI / Aoraki)
├── notebooks/              # Jupyter notebooks
│   ├── data_merging/       # Per-deer data merging notebooks
│   └── exploratory/        # Exploratory analysis & tortuosity
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

### 1. Synchronise sensor data

```python
from stag.sync.data_sync import BetterDataSync

syncer = BetterDataSync(
    deer_id="R1_D1",
    head_data=head_df,
    ear_data=ear_df,
    window_dict={"start": 0, "end": 50000},
)
syncer.run_synchronization()
```

### 2. Run clustering

```bash
python scripts/run_clustering.py \
    -t deer8 -nc 8 -ds 0 -dp 0 -rs 0 \
    -df data/clust_data_deer8.npy \
    -sd results/
```

Or submit a SLURM sweep:

```bash
sbatch slurm/run_slurm_main_clustering.sh
```

### 3. Analyse behavioural sequences

```python
from stag.analysis.label_analysis import LabelAnalyser

analyser = LabelAnalyser("results/labels.npy", fps=50)
analyser.main(cutoff=2, save_path="results/label_analysis.json")
```

## Hardware deployment

The trained nearest-centroid classifier was benchmarked on embedded
microcontrollers:

| Processor | Classifications / sec |
|-----------|----------------------|
| Arduino Uno (16 MHz ATmega328P) | 4.3 × 10⁸ |
| Arduino Micro | 4.3 × 10⁸ |
| Intel i7-13700H (single core) | 2.6 × 10⁵ |

## Citation

If you use STAG in your research, please cite:

```bibtex
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

See also [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for
details.

## Authors

- **Alexander R. H. Matthews** — Department of Zoology, University of
  Otago, Dunedin, New Zealand
- **Lindsay R. Matthews** — Matthews Research, New Zealand
- **Bart R. H. Geurten** — Department of Zoology, University of Otago,
  Dunedin, New Zealand *(corresponding author: bart.geurten@otago.ac.nz)*

## Development history

This pipeline was developed at the Department of Zoology, University of
Otago. Early development took place under
[github.com/alexrhmatthews/headshake_project](https://github.com/alexrhmatthews/headshake_project);
the codebase was reorganised and renamed to STAG for publication.
