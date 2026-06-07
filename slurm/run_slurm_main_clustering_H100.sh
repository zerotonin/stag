#!/bin/bash
# Cluster-specific batch script for the Otago Aoraki HPC; paths reflect that environment.
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"


# Adjusted command for generating filenames without requiring CUDA
filename_cmd="python ${STAG_HPC_PROJECT_DIR}/generate_filename.py"

# Base command structure for clustering, requiring CUDA
base_cmd="${STAG_HPC_CONDA_PY} ${STAG_HPC_PROJECT_DIR}/clustering_script.py"

# Root directory for data and results
data_file="${STAG_HPC_DATA_ROOT}/clust_data_deer8.npy"
result_dir="${STAG_HPC_DATA_ROOT}/cluster_results/"

# SLURM job parameters for readability, updated partition
slurm_params="--account=${STAG_HPC_USER} --partition=aoraki_gpu_H100 --nodes=1 --ntasks-per-node=1 --gpus-per-task=1 --mem=64GB"

# Order of execution for deletion sizes
declare -a execution_order=(0 50 25 10)
declare -a k_values_all=( $(seq 2 1 20) 25 30 35 40 45 50 )  # Combine all k_values

# Iterate based on the predefined order
for del_size in "${execution_order[@]}"; do
    for del_pos in $(seq 0 10 90); do
        for k in "${k_values_all[@]}"; do
            # Generate filename for checking if the job is already done
            meta_file=$($filename_cmd "$result_dir" "deer8" "$k" "$del_size" "$del_pos")
            
            # Check if the meta file already exists
            if [ ! -f "$meta_file" ]; then
                # Construct job_name with del_size, del_pos, and k
                job_name="clust_${del_size}_${del_pos}_k${k}"
                
                # Submit the job for each k value separately
                sbatch $slurm_params --job-name="$job_name" --wrap="$base_cmd -t deer5_2 -nc $k -ds $del_size -dp $del_pos -rs 0 -df $data_file -sd $result_dir"
            else
                echo "Did not cluster combination k=$k, del_size=$del_size, del_pos=$del_pos because already done."
            fi
        done
    done
done
