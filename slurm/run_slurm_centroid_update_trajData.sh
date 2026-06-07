#!/bin/bash
# Cluster-specific batch script for the Otago Aoraki HPC; paths reflect that environment.
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"


# Define paths and parameters
JSON_FILE_PATH="${STAG_HPC_DATA_ROOT}/cluster_results/deer5_2/centroid_label_info.json"
PYTHON_SCRIPT_PATH="${STAG_HPC_PROJECT_DIR}/DeerInfo.py"
ENVIRONMENT_PATH="${STAG_HPC_CPU_PY}"  # Path to the Python environment if needed

# Submit Slurm job for 'tortuosity'
sbatch <<EOF
#!/bin/bash
#SBATCH --account=${STAG_HPC_USER}
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --job-name=update_tortuosity
#SBATCH --output=tortuosity_%j.log

$ENVIRONMENT_PATH $PYTHON_SCRIPT_PATH tortuosity $JSON_FILE_PATH
EOF

# Wait for 2 seconds before submitting the next job to avoid concurrent write access
sleep 2

# Submit Slurm job for 'abs_speed_mPs'
sbatch <<EOF
#!/bin/bash
#SBATCH --account=${STAG_HPC_USER}
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --job-name=update_abs_speed_mPs
#SBATCH --output=abs_speed_mPs_%j.log

source activate $ENVIRONMENT_PATH  $PYTHON_SCRIPT_PATH abs_speed_mPs $JSON_FILE_PATH
EOF
