#!/bin/bash
# Cluster-specific batch script for the Otago Aoraki HPC; paths reflect that environment.
#
# Cluster-stability null per-(surrogate_seed, k) job.
#
# Driven by ``slurm/submit_stability_null.sh``, which fires one sbatch
# per missing (seed, k) pair with STAG_NULL_SEED / STAG_NULL_K set in
# the environment.  The submitter floods the queue with up to 180
# independent jobs and lets SLURM bin-pack them across whatever GPU
# partitions have free slots — the same opportunistic pattern Alex's
# original headshake_project clustering used.
#
# Partition list spans every GPU the lab account can submit to with
# the default normal,ondemand QOS (AllowQos=ALL on each).  H200 is
# excluded because its h200_* QOS whitelist denies us.  SLURM
# resolves the comma-separated list to whichever pool has free
# resources first.

#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100,aoraki_gpu_A100_80GB,aoraki_gpu_A100_40GB,aoraki_gpu_RTX6000,aoraki_gpu_L40,aoraki_gpu_L4_24GB,aoraki_gpu_RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --job-name=stab_null
#SBATCH --output=logs/stab_null_%j.out
#SBATCH --error=logs/stab_null_%j.err

cd "${SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source slurm/env.sh

# Fallback so this script works without editing the user's gitignored
# env.sh: if env.sh predates the surrogate-cache variable, default it
# to a sibling of the data root.
: "${STAG_HPC_NULL_SURROGATE_DIR:=${STAG_HPC_DATA_ROOT}/null_surrogates}"
export STAG_HPC_NULL_SURROGATE_DIR

set -euo pipefail
mkdir -p logs

# ─── (seed, k) come from the submitter via --export ─────────────────
SEED=${STAG_NULL_SEED:?STAG_NULL_SEED must be set by the submitter}
K=${STAG_NULL_K:?STAG_NULL_K must be set by the submitter}
SEED_PADDED=$(printf "%02d" "${SEED}")

echo "── Job ${SLURM_JOB_ID}: surrogate_seed=${SEED}  k=${K} ──"
echo "── Landed on partition=${SLURM_JOB_PARTITION}  node=$(hostname) ──"
echo "── Visible GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ──"

# ─── Workspace ────────────────────────────────────────────────────────
cd "${STAG_HPC_PROJECT_DIR}"
TABLES_DIR="results/sprint1/tables/stability_null_uniform"
mkdir -p "${TABLES_DIR}"

SURROGATE="${STAG_HPC_NULL_SURROGATE_DIR}/null_uniform_seed${SEED_PADDED}.npy"
OUT_CSV="${TABLES_DIR}/stability_null_seed${SEED_PADDED}_k${K}.csv"

# ─── Resume gate ──────────────────────────────────────────────────────
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
