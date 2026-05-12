#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  Close the k=45 hole on the Aoraki HPC cluster
#
#  The original SLURM sweep produced 50 metadata JSONs and 50 centroid
#  arrays for k=45 at every delSize ∈ {0, 10, 25, 50}, but the labels
#  .npy files were never written — the k_45 folder on HCS storage is
#  ~800 KB versus ~39 GB for every other k folder, indicating the
#  labels-saving step failed for all 200 fits in that group.
#
#  This script re-runs only the k=45 fits to produce the missing
#  labels.npy.  With the same random_state and the same data file,
#  the recomputed centroids and CH should agree with the original
#  saved ones to machine precision (cuML KMeans converges to a
#  deterministic optimum from a deterministic seed); the meta JSONs
#  and centroid arrays will simply be overwritten with bit-identical
#  values, and the missing labels will appear alongside them.
#
#  Resource use: 4 × 50 = 200 GPU-jobs.  Each fit at k=45 on H100
#  takes ~5-10 minutes; total wall-clock ~20-40 minutes if the
#  partition has capacity for 16+ concurrent jobs.
#
#  Pre-flight (see the checklist printed to stderr when this script
#  is run with no arguments):
#    1. The post-revision STAG repo at /home/geuba03p/PyProjects/stag
#    2. The pre-processed MaxAbs-scaled 6-col feature matrix at
#       /projects/sciences/zoology/geurten_lab/deer_2024/clust_data_maxabs_6col.npy
#       (produced by scripts/preprocess_clustering_data.py — matches the
#       2024 SLURM pipeline's MaxAbsScaler + col-5 ±7.99 clip exactly)
#    3. A writable cluster_results directory at the path below
#
#  Critically, the kmeans call passes --no-rescale.  The 2024
#  production pipeline applied MaxAbsScaler inside the script and
#  fed the result straight to cuML; we now do the same up-front (in
#  preprocess_clustering_data.py) and tell cuML to use the data
#  as-is.  Without --no-rescale the in-script StandardScaler would
#  re-stretch column 5 by ~25× and the resulting centroids would
#  not match the historical ones.
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

# Print the upload checklist when invoked with --check (and exit).
if [[ "${1:-}" == "--check" ]]; then
    cat >&2 <<'CHK'

  Pre-flight checklist for Aoraki (login host: aoraki-login)
  ──────────────────────────────────────────────────────────────────
   STAG package        : /home/geuba03p/PyProjects/stag/
   Data file           : /projects/sciences/zoology/geurten_lab/deer_2024/clust_data_maxabs_6col.npy
   Results root        : /projects/sciences/zoology/geurten_lab/deer_2024/cluster_results/
   Conda env           : rapids-24.02
   GPU partitions      : aoraki_gpu_H100, aoraki_gpu_L40, aoraki_gpu (A100s)

  Quick verification commands to run on aoraki-login before sbatch.

  IMPORTANT: do NOT try to import cuml or cudf on the login node —
  it has no GPU, so cuML cannot find libcuda.so.1 and its import
  machinery loops through every CUDA initialisation path before
  failing.  Test only stag-level imports here; the SLURM jobs
  exercise the GPU stack on whichever partition SLURM scheduled
  the job on (libcuda.so.1 is present on all GPU partitions).

   ls -la /home/geuba03p/PyProjects/stag/stag/clustering/kmeans.py
   ls -la /projects/sciences/zoology/geurten_lab/deer_2024/clust_data_maxabs_6col.npy

   # CPU-only import check (safe on aoraki-login):
   /home/geuba03p/miniconda3/envs/rapids-24.02/bin/python -c "
   import stag
   import pandas as pd
   print('stag', stag.__version__)
   print('stag location:', stag.__file__)
   print('pandas', pd.__version__)
   "

  If you really want a belt-and-braces GPU-stack check, grab a brief
  interactive GPU node:

   srun --account=geuba03p --partition=aoraki_gpu_H100,aoraki_gpu_L40,aoraki_gpu \
        --gpus-per-task=1 --mem=16GB --time=00:05:00 --pty bash
   /home/geuba03p/miniconda3/envs/rapids-24.02/bin/python -c "
   import cupy, cuml
   from cuml.cluster import KMeans
   print('cupy', cupy.__version__, 'cuml', cuml.__version__)
   "
   exit

CHK
    exit 0
fi

# ─── Configuration ────────────────────────────────────────────────────
CONDA_PY="/home/geuba03p/miniconda3/envs/rapids-24.02/bin/python"
STAG_PATH="/home/geuba03p/PyProjects/stag"

DATA_FILE="/projects/sciences/zoology/geurten_lab/deer_2024/clust_data_maxabs_6col.npy"
RESULT_DIR="/projects/sciences/zoology/geurten_lab/deer_2024/cluster_results/"

# Comma-separated partition list lets SLURM dispatch to whichever
# GPU queue clears first.  Order matters only as a tie-breaker —
# H100 is fastest at k=45 (~5 min), L40 ≈ A100 ≈ 8-10 min.
SLURM_PARAMS="--account=geuba03p \
              --partition=aoraki_gpu_H100,aoraki_gpu_L40,aoraki_gpu \
              --nodes=1 \
              --ntasks-per-node=1 \
              --gpus-per-task=1 \
              --mem=64GB \
              --time=01:00:00"

# Tag matches the existing tree: cluster_results/deer6raw/...
TAG="deer6raw"
K=45

# Match the original sweep's 50-position density: 0, 2, 4, ..., 98.
declare -a positions=( $(seq 0 2 98) )

# All four leave-out sizes — the silhouette / inertia panels only use
# delSize_0, but we close the hole for the full grid so the per-fit
# JSONs in delSize_{10,25,50} also gain their missing labels.
declare -a deletion_sizes=(0 50 25 10)

# Skip-if-exists guard so a partial completion can resume cleanly.
SKIPPED=0
SUBMITTED=0

# ─── Submission loop ──────────────────────────────────────────────────
for del_size in "${deletion_sizes[@]}"; do
    for del_pos in "${positions[@]}"; do
        labels_path="${RESULT_DIR}${TAG}/delSize_${del_size}/k_${K}/labels/${TAG}_labels_k${K}_delSize${del_size}_delPosP${del_pos}.npy"

        if [[ -f "$labels_path" ]]; then
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        job_name="k${K}_ds${del_size}_pos${del_pos}"
        sbatch $SLURM_PARAMS \
            --job-name="$job_name" \
            --export=ALL,PYTHONPATH="${STAG_PATH}:${PYTHONPATH:-}" \
            --wrap="${CONDA_PY} -m stag.clustering.kmeans \
                    -t ${TAG} \
                    -nc ${K} \
                    -ds ${del_size} \
                    -dp ${del_pos} \
                    -rs 0 \
                    -df ${DATA_FILE} \
                    -sd ${RESULT_DIR} \
                    --no-rescale"
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo
echo "═══════════════════════════════════════════════════════════════"
echo "  k=${K} fill-in submission summary"
echo "═══════════════════════════════════════════════════════════════"
echo "  Submitted: $SUBMITTED jobs"
echo "  Skipped (labels already present): $SKIPPED"
echo
echo "  Monitor with: squeue --user=\$USER --name=k${K}*"
echo "  Cancel all  : scancel --user=\$USER --name=k${K}*"
