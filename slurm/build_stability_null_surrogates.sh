#!/bin/bash
# Cluster-specific batch script for the Otago Aoraki HPC; paths reflect that environment.
#
# Build the uniform MaxAbs-box surrogates for the R3 cluster-stability null.
# Each task draws one Uniform(-1, 1)^6 surrogate of the same n as the real
# MaxAbs feature matrix and saves it as null_uniform_seed<SS>.npy under
# ${STAG_HPC_NULL_SURROGATE_DIR}.
#
# Submit with:
#   sbatch --array=0-9 slurm/build_stability_null_surrogates.sh
#
# 10 tasks = 10 surrogate seeds.  Each .npy is ~4.6 GB (float32 × 204.5 M × 6).
# Total cache footprint: ~46 GB.
#
# Idempotent: re-running re-uses any existing .npy.

#SBATCH --account=geuba03p
#SBATCH --partition=aoraki
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=stab_null_build
#SBATCH --output=logs/stab_null_build_%A_%a.out
#SBATCH --error=logs/stab_null_build_%A_%a.err

cd "${SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source slurm/env.sh

# Fallback so this script works on Aoraki without editing the user's
# gitignored env.sh: if env.sh predates the surrogate-cache variable,
# default it to a sibling of the data root.
: "${STAG_HPC_NULL_SURROGATE_DIR:=${STAG_HPC_DATA_ROOT}/null_surrogates}"
export STAG_HPC_NULL_SURROGATE_DIR

set -euo pipefail
mkdir -p logs

cd "${STAG_HPC_PROJECT_DIR}"

SEED=${SLURM_ARRAY_TASK_ID}
SURROGATE_DIR="${STAG_HPC_NULL_SURROGATE_DIR}"
mkdir -p "${SURROGATE_DIR}"

echo "── Task ${SLURM_ARRAY_TASK_ID}: building uniform surrogate seed=${SEED} ──"
echo "── Output dir: ${SURROGATE_DIR} ──"

"${STAG_HPC_CPU_PY}" -m scripts.build_null_surrogate \
    --seed "${SEED}" \
    --output-dir "${SURROGATE_DIR}" \
    --reference "${STAG_HPC_DATA_ROOT}/clust_data_maxabs_6col.npy"

echo "── Done. ──"
