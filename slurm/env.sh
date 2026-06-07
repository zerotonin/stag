# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — slurm/env.sh                                             ║
# ║  « shared environment for the Otago Aoraki HPC batch scripts »   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Every script in this directory is cluster-specific: it was      ║
# ║  written to run under SLURM on the University of Otago "Aoraki"  ║
# ║  HPC, where the package was installed under                      ║
# ║  ~/PyProjects/stag with a RAPIDS conda environment and the deer  ║
# ║  data archive lived at                                           ║
# ║  /projects/sciences/zoology/geurten_lab/deer_2024.  The scripts  ║
# ║  are committed for provenance — they document how the 2024       ║
# ║  clustering sweep and the GPS / tortuosity batch jobs were       ║
# ║  produced — not as ready-to-run examples on an arbitrary cluster.║
# ║                                                                  ║
# ║  Source this file at the top of every slurm/*.sh; export each    ║
# ║  variable below with the canonical default but allow the         ║
# ║  caller's environment to override.                               ║
# ╚══════════════════════════════════════════════════════════════════╝

# Aoraki account / home directory owner (most jobs ran under geuba03p;
# a small number of legacy sync/tortuosity jobs ran as matal178 — set
# STAG_HPC_USER=matal178 to switch).
: "${STAG_HPC_USER:=geuba03p}"

# Where the STAG package is installed on Aoraki.
: "${STAG_HPC_PROJECT_DIR:=/home/${STAG_HPC_USER}/PyProjects/stag}"

# Path to the RAPIDS conda Python that the GPU clustering jobs use.
: "${STAG_HPC_CONDA_PY:=/home/${STAG_HPC_USER}/miniconda3/envs/rapids-24.02/bin/python}"

# Plain CPU conda Python for tortuosity / sync / DB-build jobs.
: "${STAG_HPC_CPU_PY:=/home/${STAG_HPC_USER}/miniconda3/envs/deer_project_2/bin/python}"

# Lab project archive on Aoraki — anchor for inputs and outputs.
: "${STAG_HPC_DATA_ROOT:=/projects/sciences/zoology/geurten_lab/deer_2024}"

# Sibling files_extracted tree on Aoraki (raw inputs + sync outputs).
: "${STAG_HPC_FILES_EXTRACTED:=/projects/sciences/zoology/geurten_lab/files_extracted}"

# Primary SQLite database (cluster_labels + GPS + video tables).
: "${STAG_HPC_DATA_DB:=${STAG_HPC_DATA_ROOT}/deer_data_gps.db}"

# Sync-pipeline data + DB folder used by make_deer_db.sh.
: "${STAG_HPC_AORAKI_MERGED_SIGNALS_V2:=${STAG_HPC_FILES_EXTRACTED}/new_file_sync/data/}"
: "${STAG_HPC_AORAKI_DB_FOLDER:=${STAG_HPC_FILES_EXTRACTED}/new_file_sync/data}"

# Lab Deer-codes CSV used by the deer-wise dispatchers.
: "${STAG_HPC_DEER_CODES:=${STAG_HPC_DATA_ROOT}/Deer_codes.csv}"

: "${STAG_HPC_ALEX_HOME:=/home/matal178}"
: "${STAG_HPC_ALEX_PROJECT_DIR:=${STAG_HPC_ALEX_HOME}/PyProjects/stag}"

export STAG_HPC_USER STAG_HPC_PROJECT_DIR STAG_HPC_CONDA_PY STAG_HPC_CPU_PY
export STAG_HPC_DATA_ROOT STAG_HPC_FILES_EXTRACTED STAG_HPC_DATA_DB
export STAG_HPC_AORAKI_MERGED_SIGNALS_V2 STAG_HPC_AORAKI_DB_FOLDER
export STAG_HPC_DEER_CODES
export STAG_HPC_ALEX_HOME STAG_HPC_ALEX_PROJECT_DIR
