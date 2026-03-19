#!/bin/bash
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=vid_availability  # Unique job name using the h5 file basename

~/miniconda3/envs/deer_project_2/bin/python /home/geuba03p/PyProjects/headshake_project/DeerInfo.py
