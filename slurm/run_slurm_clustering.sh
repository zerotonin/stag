#!/bin/bash
#SBATCH --job-name=clust_go1
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64GB

 /home/geuba03p/miniconda3/envs/rapids-24.02/bin/python /home/geuba03p/PyProjects/headshake_project/clustering_script.py -t deer8 -nc 2 -ds 50 -dp 2 -rs 0 -df /projects/sciences/zoology/geurten_lab/deer_2024/clust_data_deer8.npy -sd /projects/sciences/zoology/geurten_lab/deer_2024/cluster_results/