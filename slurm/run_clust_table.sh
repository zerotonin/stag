#!/bin/bash
#SBATCH --job-name=clust_tab
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G   
/home/geuba03p/miniconda3/envs/velvet/bin/python /home/geuba03p/PyProjects/headshake_project/DeerInfo.py
