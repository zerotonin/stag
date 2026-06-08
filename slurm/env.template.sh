# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — slurm/env.template.sh                                    ║
# ║  « scaffold for the per-cluster SLURM environment file »         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Every script in this directory expects a sibling file at        ║
# ║  ``slurm/env.sh`` (gitignored, per-cluster, never committed)     ║
# ║  that defines the absolute paths the SLURM jobs need.            ║
# ║                                                                  ║
# ║  Bootstrap once per cluster:                                     ║
# ║    cp slurm/env.template.sh slurm/env.sh                         ║
# ║    $EDITOR slurm/env.sh   # replace every <placeholder>          ║
# ║                                                                  ║
# ║  Variable names start with ``STAG_HPC_`` to keep them generic    ║
# ║  across HPC sites - "HPC" is the abbreviation, the values are    ║
# ║  what differ between clusters.  Lab provenance: the original     ║
# ║  STAG team ran on the University of Otago Aoraki HPC SLURM       ║
# ║  cluster, so the variable names and the workflow expectations    ║
# ║  reflect that environment, but the same template works for any   ║
# ║  SLURM-based HPC that hosts the deer_2024 archive and a RAPIDS   ║
# ║  conda environment.                                              ║
# ║                                                                  ║
# ║  Override any value via the calling environment - the ``:=``     ║
# ║  assignments only fire when the variable is unset.               ║
# ╚══════════════════════════════════════════════════════════════════╝

# ─── HPC identity ────────────────────────────────────────────────────
# Account / username on the cluster (SLURM `--account` and home dir).
: "${STAG_HPC_USER:=<your cluster username>}"

# Where the STAG package is checked out on the cluster.
: "${STAG_HPC_PROJECT_DIR:=<absolute path to STAG checkout, e.g. /home/<user>/PyProjects/stag>}"

# ─── Python environments ─────────────────────────────────────────────
# RAPIDS-enabled conda Python for the GPU clustering jobs.
: "${STAG_HPC_CONDA_PY:=<absolute path to the RAPIDS conda Python, e.g. /home/<user>/miniconda3/envs/rapids-24.02/bin/python>}"

# Plain CPU conda Python for tortuosity / sync / DB-build / silhouette jobs.
: "${STAG_HPC_CPU_PY:=<absolute path to a CPU conda Python with scipy + sklearn>}"

# ─── Lab data archive on the cluster ─────────────────────────────────
# Project root for the deer_2024 archive (contains clust_data_*.npy,
# cluster_results/, deer_data_gps.db, ...).
: "${STAG_HPC_DATA_ROOT:=<absolute path to deer_2024 archive on the cluster>}"

# Sibling ``files_extracted`` tree (raw CSV inputs + sync output dirs).
: "${STAG_HPC_FILES_EXTRACTED:=<absolute path to files_extracted/ on the cluster>}"

# Cache directory for cluster-stability null surrogates (see
# ``stag.analysis.stability_null`` / ``scripts/build_null_surrogate.py``).
# Each draw is a 4.6 GB float32 .npy at the same n as the MaxAbs feature
# matrix; ten seeds = ~46 GB.  Defaults to a sibling of the data root so
# the surrogates live on the same fast filesystem as the real input.
: "${STAG_HPC_NULL_SURROGATE_DIR:=${STAG_HPC_DATA_ROOT}/null_surrogates}"

# Primary SQLite database (cluster_labels + GPS + video tables).
: "${STAG_HPC_DATA_DB:=${STAG_HPC_DATA_ROOT}/deer_data_gps.db}"

# Sync-pipeline data + DB folder used by make_deer_db.sh.
: "${STAG_HPC_MERGED_SIGNALS_V2:=${STAG_HPC_FILES_EXTRACTED}/new_file_sync/data/}"
: "${STAG_HPC_DB_FOLDER:=${STAG_HPC_FILES_EXTRACTED}/new_file_sync/data}"

# Lab Deer-codes CSV used by the deer-wise dispatchers.
: "${STAG_HPC_DEER_CODES:=${STAG_HPC_DATA_ROOT}/Deer_codes.csv}"

# ─── Co-author home directory (optional, for re-running legacy sync) ─
: "${STAG_HPC_COAUTHOR_HOME:=<absolute path to co-author cluster home, or leave the placeholder if not needed>}"
: "${STAG_HPC_COAUTHOR_PROJECT_DIR:=${STAG_HPC_COAUTHOR_HOME}/PyProjects/stag}"

# ─── Mirror into the Python-side env vars ────────────────────────────
# Sourcing this file is sufficient for ``stag.constants`` /
# ``stag.local_paths`` on the cluster: each Python-side variable falls
# back to its STAG_HPC_* counterpart so a separate ``local_paths.json``
# does not have to be maintained on the HPC.
: "${STAG_DATA_DIR:=${STAG_HPC_DATA_ROOT}}"
: "${STAG_DEER_DB_URL:=sqlite:///${STAG_HPC_DATA_DB}}"

# ─── Export everything for the SLURM job environment ─────────────────
export STAG_HPC_USER STAG_HPC_PROJECT_DIR STAG_HPC_CONDA_PY STAG_HPC_CPU_PY
export STAG_HPC_DATA_ROOT STAG_HPC_FILES_EXTRACTED STAG_HPC_DATA_DB
export STAG_HPC_NULL_SURROGATE_DIR
export STAG_HPC_MERGED_SIGNALS_V2 STAG_HPC_DB_FOLDER STAG_HPC_DEER_CODES
export STAG_HPC_COAUTHOR_HOME STAG_HPC_COAUTHOR_PROJECT_DIR
export STAG_DATA_DIR STAG_DEER_DB_URL
