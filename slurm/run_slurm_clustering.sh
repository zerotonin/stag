#!/bin/bash
# Cluster-specific batch script for the Otago Aoraki HPC; paths reflect that environment.
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

#SBATCH --job-name=clust_go1
#SBATCH --account=${STAG_HPC_USER}
#SBATCH --partition=aoraki_gpu_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64GB

 ${STAG_HPC_CONDA_PY} ${STAG_HPC_PROJECT_DIR}/clustering_script.py -t deer8 -nc 2 -ds 50 -dp 2 -rs 0 -df ${STAG_HPC_DATA_ROOT}/clust_data_deer8.npy -sd ${STAG_HPC_DATA_ROOT}/cluster_results/
