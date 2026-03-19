#!/bin/bash
#SBATCH --account=account_name
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=deer_checking  # Set a unique job name for each analysis
~/miniconda3/envs/deer_project_2/bin/python /home/matal178/PyProjects/headshake_project/headshake_project/check_synced_data.py


