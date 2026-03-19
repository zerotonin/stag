# Path to the CSV file containing deer and repetition numbers
DEER_CODES_FILE="./Deer_Codes_fromDB.csv"

while IFS=, read -r deer_number repetition_number; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB  # Updated memory requirement
#SBATCH --cpus-per-task=1
#SBATCH --job-name=vidAvail_${deer_number}_${repetition_number}
~/miniconda3/envs/deer_project_2/bin/python /home/geuba03p/PyProjects/headshake_project/DeerInfo.py "$deer_number" "$repetition_number" "/projects/sciences/zoology/geurten_lab/deer_2024/vid_avail_cache/"
EOF
done < "$DEER_CODES_FILE"