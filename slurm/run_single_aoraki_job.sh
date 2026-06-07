#!/bin/bash
# Cluster-specific batch script for the Otago Aoraki HPC; paths reflect that environment.
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

#SBATCH --account=${STAG_HPC_USER}
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=vid_availability  # Unique job name using the h5 file basename

~/miniconda3/envs/deer_project_2/bin/python ${STAG_HPC_PROJECT_DIR}/DeerInfo.py
