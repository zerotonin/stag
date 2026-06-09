#!/bin/bash
# Cluster-specific batch wrapper for the Otago Aoraki HPC; paths reflect that environment.
#
# Opportunistic per-task stability-null submitter.  Scans for missing
# (surrogate_seed, k) outputs under
# results/sprint1/tables/stability_null_uniform/ and fires one sbatch
# per missing pair via slurm/stability_null_single.sh.
#
# Each sbatch carries a comma-separated --partition= list spanning the
# whole pool of GPU partitions this account can submit to.  SLURM
# bin-packs across H100 / A100 / L40 / RTX6000 / L4 / RTX3090 as slots
# free up — the same flood-and-let-SLURM-sort pattern Alex's original
# headshake_project clustering scripts used.
#
# Idempotent: re-run as many times as you like; the per-task resume
# gate in stability_null_single.sh skips finished CSVs.
#
# Usage (run from the STAG project root):
#   bash slurm/submit_stability_null.sh         # auto-submit missing
#   bash slurm/submit_stability_null.sh --dry   # preview without firing

set -euo pipefail

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
# Full integer grid k = 2 .. 20 plus sparse high-k anchors 25, 35, 45, 50.
K_VALUES=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 35 45 50)

# Partition pools.
#
# aoraki_gpu_L4_24GB is excluded from both pools.  aoraki29's seven L4
# GPUs are sharded (10 shards per GPU), so concurrent jobs share a
# 24 GB VRAM budget; cuML's rmm allocator collides with whoever else
# is on the GPU and OOMs even at k = 2.  Confirmed by the first
# batch: every one of the 29 first-run failures (k = 2 .. 50) was
# the same std::bad_alloc inside kmeans.fit() on aoraki29.
#
# RTX nodes (RTX6000, RTX3090) are excluded for k > 25 because the
# per-fit time at high k pushes them into the 02:30:00 walltime
# cliff; H100 / A100 / L40 carry the high-k load.
PART_LOW_K="aoraki_gpu_H100,aoraki_gpu_A100_80GB,aoraki_gpu_A100_40GB,aoraki_gpu_RTX6000,aoraki_gpu_L40,aoraki_gpu_RTX3090"
PART_HIGH_K="aoraki_gpu_H100,aoraki_gpu_A100_80GB,aoraki_gpu_A100_40GB,aoraki_gpu_L40"

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

# ─── Mode select ─────────────────────────────────────────────────────
DRY=false
if [[ "${1:-}" == "--dry" ]]; then DRY=true; fi

# ─── Enumerate missing (seed, k) and fire ────────────────────────────
SUBMITTED=0
SKIPPED=0
PLANNED=()

for seed in "${SEEDS[@]}"; do
    seed_padded=$(printf "%02d" "${seed}")
    for k in "${K_VALUES[@]}"; do
        out_csv="${TABLES_DIR}/stability_null_seed${seed_padded}_k${k}.csv"
        if [[ -s "${out_csv}" ]] && [[ "$(wc -l < "${out_csv}")" -ge 2 ]]; then
            (( SKIPPED += 1 ))
            continue
        fi
        PLANNED+=("seed=${seed} k=${k}")
        if $DRY; then
            echo "  would submit  seed=${seed_padded}  k=${k}  → ${out_csv}"
            continue
        fi
        # k > 25 keeps off the RTX pool; everything else uses the full list.
        if (( k > 25 )); then
            partition="${PART_HIGH_K}"
        else
            partition="${PART_LOW_K}"
        fi
        sbatch \
            --partition="${partition}" \
            --export=ALL,STAG_NULL_SEED="${seed}",STAG_NULL_K="${k}" \
            --job-name="stab_null_s${seed_padded}_k${k}" \
            slurm/stability_null_single.sh
        (( SUBMITTED += 1 ))
    done
done

TOTAL=$(( ${#SEEDS[@]} * ${#K_VALUES[@]} ))
echo
echo "Grid total       : ${TOTAL}"
echo "Already complete : ${SKIPPED}"
if $DRY; then
    echo "Would submit     : ${#PLANNED[@]}"
    echo "(dry run — no sbatch fired)"
else
    echo "Submitted        : ${SUBMITTED}"
    echo
    echo "k <= 25 lands on : ${PART_LOW_K}"
    echo "k >  25 lands on : ${PART_HIGH_K}  (RTX excluded)"
    echo
    echo "Watch with:  squeue -u \$USER -o \"%i %P %j %T %M %R\""
    echo "             squeue -u \$USER --noheader | wc -l   # remaining count"
fi
