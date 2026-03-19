#!/bin/bash

# Define paths and parameters
JSON_FILE_PATH="/projects/sciences/zoology/geurten_lab/deer_2024/cluster_results/deer5_2/centroid_label_info.json"
PYTHON_SCRIPT_PATH="/home/geuba03p/PyProjects/headshake_project/DeerInfo.py"
ENVIRONMENT_PATH="/home/geuba03p/miniconda3/envs/deer_project_2/bin/python"  # Path to the Python environment if needed

# Submit Slurm job for 'tortuosity'
sbatch <<EOF
#!/bin/bash
#SBATCH --account=geuba03p
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
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --job-name=update_abs_speed_mPs
#SBATCH --output=abs_speed_mPs_%j.log

source activate $ENVIRONMENT_PATH  $PYTHON_SCRIPT_PATH abs_speed_mPs $JSON_FILE_PATH
EOF
