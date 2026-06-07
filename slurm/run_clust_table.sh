#!/bin/bash
# Cluster-specific batch script for the Otago Aoraki HPC; paths reflect that environment.
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

#SBATCH --job-name=clust_tab
#SBATCH --account=${STAG_HPC_USER}
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G   
${STAG_HPC_CPU_PY} ${STAG_HPC_PROJECT_DIR}/DeerInfo.py
