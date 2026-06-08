#!/bin/bash
# Cluster-specific batch wrapper for the Otago Aoraki HPC; paths reflect that environment.
#
# Smart stability-null submitter — scans
# results/sprint1/tables/stability_null_uniform/stability_null_seed<SS>_k<KK>.csv
# to find the (surrogate_seed, k) pairs that still need data, builds a
# sparse SLURM array spec containing only those task IDs, and sbatches
# them with the four-GPU concurrency throttle (%4).
#
# Pre-flight: every uniform surrogate must already be on disk under
# ${STAG_HPC_NULL_SURROGATE_DIR}.  Build them first with:
#   sbatch slurm/build_stability_null_surrogates.sh
#
# Usage (run from the STAG project root):
#   bash slurm/submit_stability_null.sh         # auto-submit
#   bash slurm/submit_stability_null.sh --dry   # show array spec only

set -euo pipefail

# Resolve the project root from the script location so this also
# works when called from elsewhere.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source slurm/env.sh

# Fallback so this wrapper works without editing the user's gitignored
# env.sh: if env.sh predates the surrogate-cache variable, default it
# to a sibling of the data root.
: "${STAG_HPC_NULL_SURROGATE_DIR:=${STAG_HPC_DATA_ROOT}/null_surrogates}"
export STAG_HPC_NULL_SURROGATE_DIR

SEEDS=(0 1 2 3 4 5 6 7 8 9)
K_VALUES=(2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 35 45 50)
N_K=${#K_VALUES[@]}

TABLES_DIR="results/sprint1/tables/stability_null_uniform"
mkdir -p "${TABLES_DIR}"

# ─── Pre-flight: every surrogate present? ────────────────────────────
SURROGATE_DIR="${STAG_HPC_NULL_SURROGATE_DIR}"
MISSING_SURROGATES=()
for seed in "${SEEDS[@]}"; do
    seed_padded=$(printf "%02d" "${seed}")
    surrogate="${SURROGATE_DIR}/null_uniform_seed${seed_padded}.npy"
    if [[ ! -s "${surrogate}" ]]; then
        MISSING_SURROGATES+=("${seed}")
    fi
done
if (( ${#MISSING_SURROGATES[@]} > 0 )); then
    echo "ERROR: missing surrogates under ${SURROGATE_DIR} for seeds:"
    echo "  ${MISSING_SURROGATES[*]}"
    echo
    echo "Build them first:"
    echo "  sbatch slurm/build_stability_null_surrogates.sh"
    exit 1
fi
echo "All 10 surrogates present under ${SURROGATE_DIR}."

# ─── Scan for missing (seed, k) outputs ──────────────────────────────
MISSING_TASK_IDS=()
TOTAL=$(( ${#SEEDS[@]} * N_K ))

for (( task_id=0; task_id<TOTAL; task_id++ )); do
    seed_idx=$(( task_id / N_K ))
    k_idx=$(( task_id % N_K ))
    seed=${SEEDS[$seed_idx]}
    k=${K_VALUES[$k_idx]}
    seed_padded=$(printf "%02d" "${seed}")
    out_csv="${TABLES_DIR}/stability_null_seed${seed_padded}_k${k}.csv"
    if [[ -s "${out_csv}" ]] && [[ "$(wc -l < "${out_csv}")" -ge 2 ]]; then
        echo "  ✓ task ${task_id}  (seed=${seed}, k=${k})  → ${out_csv}"
    else
        echo "  ✗ task ${task_id}  (seed=${seed}, k=${k})  → MISSING"
        MISSING_TASK_IDS+=("${task_id}")
    fi
done

N_MISSING=${#MISSING_TASK_IDS[@]}
echo
echo "Summary: ${N_MISSING} of ${TOTAL} task outputs are missing."

if (( N_MISSING == 0 )); then
    echo "Nothing to submit — the (seed, k) grid is complete."
    echo "Run scripts/merge_stability_null.py to assemble the band."
    exit 0
fi

ARRAY_SPEC=$( IFS=, ; echo "${MISSING_TASK_IDS[*]}" )

echo "Array spec to submit: --array=${ARRAY_SPEC}%4"
echo "(four concurrent GPU tasks — matches the four-GPU 24 h plan)"

if [[ "${1:-}" == "--dry" ]]; then
    echo "(dry run — not firing sbatch)"
    exit 0
fi

sbatch --array="${ARRAY_SPEC}%4" slurm/stability_null_array.sh
echo
echo "Watch with:  squeue -u \$USER"
