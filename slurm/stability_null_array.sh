#!/bin/bash
# Cluster-specific batch script for the Otago Aoraki HPC; paths reflect that environment.
#
# Cluster-stability null SLURM array — runs one (surrogate_seed, k) block
# per array task.  Each task loads the matching uniform surrogate, runs
# 20 GPU k-means fits at the requested k, and emits a CSV with the per-fit
# Hungarian-matched centroid drift.
#
# Submit with:
#   sbatch --array=0-179%4 slurm/stability_null_array.sh
#
# 180 tasks = 10 surrogate seeds × 18 k-values.  The k-grid is fine
# (step 1) over k = 2 .. 10, medium (step 2) over k = 12 .. 20, and
# sparse over k = 25, 35, 45, 50 — favours resolution where the
# elbow sits.  The %4 throttle caps simultaneous GPU jobs at four,
# matching the four-GPU plan.
#
# Partition is aoraki_gpu_H200 (aoraki44, 8 GPUs, 144 GB each).
# The H200 pool is essentially uncontested on Aoraki, so the %4
# throttle below grabs four GPUs on one node with no queue wait.
#
# Empirical per-fit median on the slower L40 partition (from the
# existing delSize_0 metadata JSONs):
#   k =  2..10  -> ~80 s,
#   k = 12..20  -> ~108 s,
#   k = 25..50  -> ~140 s
# H200 is roughly 3x faster than L40 on this bandwidth-bound work
# (4.8 TB/s vs 0.86 TB/s memory bandwidth), so 20 fits per task
# land in ~15 min nominal at high k.  --time=03:30:00 keeps the
# ceiling safe even on the long-tail outliers in the archive (a
# few fits per k bucket hit ~25 min for queue / init reasons).
#
# Predicted wall-clock: ~9 h ideal / ~12 h realistic on 4 H200s.
# Fallback: if the H200 node is unexpectedly down, swap to
# aoraki_gpu_H100 (~13 h / 16 h) or aoraki_gpu_L40 (~25 h / 30 h);
# the resume gate ensures only missing tasks run on the re-fire.
#
# Idempotent: per-task resume gate skips tasks whose CSV already exists.

#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --job-name=stab_null
#SBATCH --output=logs/stab_null_%A_%a.out
#SBATCH --error=logs/stab_null_%A_%a.err

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

# ─── Index into the (surrogate_seed, k) grid ─────────────────────────
SEEDS=(0 1 2 3 4 5 6 7 8 9)
K_VALUES=(2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 35 45 50)
N_K=${#K_VALUES[@]}

SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_K ))
K_IDX=$(( SLURM_ARRAY_TASK_ID % N_K ))

SEED=${SEEDS[$SEED_IDX]}
K=${K_VALUES[$K_IDX]}
SEED_PADDED=$(printf "%02d" "${SEED}")

echo "── Task ${SLURM_ARRAY_TASK_ID}: surrogate_seed=${SEED}  k=${K} ──"

# ─── Workspace ────────────────────────────────────────────────────────
cd "${STAG_HPC_PROJECT_DIR}"
TABLES_DIR="results/sprint1/tables/stability_null_uniform"
mkdir -p "${TABLES_DIR}"

SURROGATE="${STAG_HPC_NULL_SURROGATE_DIR}/null_uniform_seed${SEED_PADDED}.npy"
OUT_CSV="${TABLES_DIR}/stability_null_seed${SEED_PADDED}_k${K}.csv"

# ─── Resume gate ─────────────────────────────────────────────────────
if [[ -s "${OUT_CSV}" ]] && [[ "$(wc -l < "${OUT_CSV}")" -ge 2 ]]; then
    echo "── Already done: ${OUT_CSV} exists with data rows, skipping. ──"
    exit 0
fi

# ─── Pre-flight: surrogate must exist ────────────────────────────────
if [[ ! -s "${SURROGATE}" ]]; then
    echo "ERROR: surrogate ${SURROGATE} not found."
    echo "Build it first with:"
    echo "  sbatch slurm/build_stability_null_surrogates.sh"
    exit 2
fi

# ─── Fire the GPU k-means block ──────────────────────────────────────
# kmeans-seed-base spreads (seed × 1000) so random states never collide
# across surrogate seeds even if n-fits is bumped later.
"${STAG_HPC_CONDA_PY}" -m scripts.run_stability_null_block \
    --surrogate "${SURROGATE}" \
    --surrogate-seed "${SEED}" \
    --k "${K}" \
    --n-fits 20 \
    --kmeans-seed-base "$(( SEED * 1000 ))" \
    --ch-subsample 200000 \
    --output-csv "${OUT_CSV}"

echo "── Wrote ${OUT_CSV} ──"
