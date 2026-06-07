#!/bin/bash
# Cluster-specific batch script for the Otago Aoraki HPC; paths reflect that environment.
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

# Path to the CSV file containing deer and repetition numbers
DEER_CODES_FILE="./Deer_Codes_fromDB.csv"

while IFS=, read -r deer_number repetition_number; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=${STAG_HPC_USER}
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB  # Updated memory requirement
#SBATCH --cpus-per-task=1
#SBATCH --job-name=vidAvail_${deer_number}_${repetition_number}
~/miniconda3/envs/deer_project_2/bin/python ${STAG_HPC_PROJECT_DIR}/DeerInfo.py "$deer_number" "$repetition_number" "${STAG_HPC_DATA_ROOT}/vid_avail_cache/"
EOF
done < "$DEER_CODES_FILE"
