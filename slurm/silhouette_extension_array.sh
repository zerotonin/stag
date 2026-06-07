#!/bin/bash
# Cluster-specific batch script for the Otago Aoraki HPC; paths reflect that environment.
#
# Silhouette-extension job array.  Computes one stratified silhouette
# at a single (delSize, k) pair per array task and emits a per-task
# single-row CSV.  After every task finishes, merge with the existing
# k <= 20 CSV via scripts/merge_silhouette_extension.py.
#
# Submit with:
#   sbatch --array=0-23 slurm/silhouette_extension_array.sh
#
# 24 tasks = 4 delSizes × 6 k-values (25, 30, 35, 40, 45, 50).
# Each task writes results/sprint1/tables/silhouette_ext_<delSize>_<k>.csv
# under STAG_HPC_PROJECT_DIR.

#SBATCH --account=geuba03p
#SBATCH --partition=aoraki
#SBATCH --array=0-23
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=silh_ext
#SBATCH --output=logs/silh_ext_%A_%a.out
#SBATCH --error=logs/silh_ext_%A_%a.err

source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

set -euo pipefail
mkdir -p logs

# ─── Index into the (delSize, k) grid ────────────────────────────────
DELSIZES=(0 10 25 50)
K_VALUES=(25 30 35 40 45 50)
N_K=${#K_VALUES[@]}

DELSIZE_IDX=$(( SLURM_ARRAY_TASK_ID / N_K ))
K_IDX=$(( SLURM_ARRAY_TASK_ID % N_K ))

DELSIZE=${DELSIZES[$DELSIZE_IDX]}
K=${K_VALUES[$K_IDX]}

echo "── Task ${SLURM_ARRAY_TASK_ID}: delSize=${DELSIZE}  k=${K} ──"

# ─── Workspace ───────────────────────────────────────────────────────
cd "${STAG_HPC_PROJECT_DIR}"
TABLES_DIR="results/sprint1/tables"
mkdir -p "${TABLES_DIR}"

OUT_CSV="${TABLES_DIR}/silhouette_ext_delSize${DELSIZE}_k${K}.csv"

# ─── Drive the single-task silhouette pass ──────────────────────────
"${STAG_HPC_CPU_PY}" -m scripts.silhouette_elbow_4reductions \
    --meta-dir "${STAG_HPC_DATA_ROOT}/cluster_results/deer6raw" \
    --data-file "${STAG_HPC_DATA_ROOT}/clust_data_maxabs_6col.npy" \
    --inertia-csv results/figures/figure2_internal_metrics_4reductions.csv \
    --output-dir results/sprint1 \
    --silhouette-per-cluster 5000 \
    --silhouette-repeats 5 \
    --only-delsize "${DELSIZE}" \
    --only-k "${K}" \
    --csv-only \
    --silhouette-csv-out "${OUT_CSV}"

echo "── Wrote ${OUT_CSV} ──"
