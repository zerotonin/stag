#!/bin/bash
#SBATCH --job-name=clust_go1
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB

 /home/geuba03p/miniconda3/envs/rapids-24.02/bin/python /home/geuba03p/PyProjects/headshake_project/getMuSigmaForZscore.py