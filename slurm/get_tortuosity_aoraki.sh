#!/bin/bash
# Path to the CSV file containing deer codes
DEER_CODES_FILE="/projects/sciences/zoology/geurten_lab/Deer_codes.csv"

while IFS= read -r line; do
    deer_code=$(echo "$line" | tr -d '\r\n')
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=account_name
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=deer_analysis${deer_code}  # Set a unique job name for each analysis
~/miniconda3/envs/deer_project_2/bin/python /home/matal178/PyProjects/headshake_project/headshake_project/get_tortuosity_for_aoraki.py "$deer_code" "cluster_paths_2"
EOF
done < "$DEER_CODES_FILE"



