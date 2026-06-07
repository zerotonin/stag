#!/bin/bash
# Cluster-specific batch script for the Otago Aoraki HPC; paths reflect that environment.
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

#SBATCH --account=account_name
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=deer_checking  # Set a unique job name for each analysis
~/miniconda3/envs/deer_project_2/bin/python ${STAG_HPC_ALEX_PROJECT_DIR}/scripts/check_synced_data.py


