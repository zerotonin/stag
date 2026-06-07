#!/bin/bash
# Cluster-specific batch script for the Otago Aoraki HPC; paths reflect that environment.
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

python_interpreter="${STAG_HPC_CPU_PY}"
data_folder="${STAG_HPC_AORAKI_MERGED_SIGNALS_V2}"
python_script="${STAG_HPC_PROJECT_DIR}/DeerInfo.py"
engine_url="sqlite:///${STAG_HPC_AORAKI_DB_FOLDER}/deer_data.db"

for filename in "${data_folder}"*.h5; do
    base_filename=$(basename "${filename}" .h5)
    job_name=$(echo "${base_filename}" | head -c 10) 
    # Truncate or hash the filename if it is too long for a Slurm job name
    # Example: job_name=$(echo "${base_filename}" | head -c 10) 
    echo $python_interpreter $python_script $filename $engine_url
    # Submit a Slurm job for this file
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=DA_${job_name}  # Unique job name 
#SBATCH --account=account_name
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/desired/output/directory/${base_filename}.out  # Specify the directory for output
#SBATCH --mem=2G                        # Adjust as needed

$python_interpreter $python_script $filename $engine_url
EOF
done
