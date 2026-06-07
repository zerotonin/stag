#!/bin/bash
# Cluster-specific batch wrapper for the Otago Aoraki HPC; paths reflect that environment.
#
# Smart silhouette-extension submitter — scans
# results/sprint1/tables/silhouette_ext_delSize<X>_k<Y>.csv to find
# the (delSize, k) pairs that still need data, builds a sparse SLURM
# array spec containing only those task IDs, and sbatches it.
#
# Idempotent: run it as many times as you like.  After every batch
# completes, run it again — it will queue only the pairs that did
# not produce a CSV in the previous run (OOM, timeout, sensor
# tantrum, whatever).
#
# Usage (run from the STAG project root):
#   bash slurm/submit_silhouette_extension.sh         # auto-submit
#   bash slurm/submit_silhouette_extension.sh --dry   # show what
#                                                     # would be sub-
#                                                     # mitted without
#                                                     # firing sbatch

set -euo pipefail

# Resolve the project root from the script location so this also
# works when called from elsewhere.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source slurm/env.sh

DELSIZES=(0 10 25 50)
K_VALUES=(25 30 35 40 45 50)
N_K=${#K_VALUES[@]}

TABLES_DIR="results/sprint1/tables"
mkdir -p "${TABLES_DIR}"

# ─── Scan for missing (delSize, k) outputs ──────────────────────────
MISSING_TASK_IDS=()
TOTAL=$(( ${#DELSIZES[@]} * N_K ))

for (( task_id=0; task_id<TOTAL; task_id++ )); do
    delsize_idx=$(( task_id / N_K ))
    k_idx=$(( task_id % N_K ))
    delsize=${DELSIZES[$delsize_idx]}
    k=${K_VALUES[$k_idx]}
    out_csv="${TABLES_DIR}/silhouette_ext_delSize${delsize}_k${k}.csv"
    if [[ -s "${out_csv}" ]] && [[ "$(wc -l < "${out_csv}")" -ge 2 ]]; then
        echo "  ✓ task ${task_id}  (delSize=${delsize}, k=${k})  → ${out_csv}"
    else
        echo "  ✗ task ${task_id}  (delSize=${delsize}, k=${k})  → MISSING"
        MISSING_TASK_IDS+=("${task_id}")
    fi
done

N_MISSING=${#MISSING_TASK_IDS[@]}
echo
echo "Summary: ${N_MISSING} of ${TOTAL} task outputs are missing."

if (( N_MISSING == 0 )); then
    echo "Nothing to submit — the (delSize, k) grid is complete."
    echo "Run scripts/merge_silhouette_extension.py to assemble the figure."
    exit 0
fi

# Build the SLURM array spec from the list (e.g. "1,2,3,5,7-9").
ARRAY_SPEC=$( IFS=, ; echo "${MISSING_TASK_IDS[*]}" )

echo "Array spec to submit: --array=${ARRAY_SPEC}"

if [[ "${1:-}" == "--dry" ]]; then
    echo "(dry run — not firing sbatch)"
    exit 0
fi

# ─── Fire the sparse array ───────────────────────────────────────────
sbatch --array="${ARRAY_SPEC}" slurm/silhouette_extension_array.sh
echo
echo "Watch with:  squeue -u \$USER"
