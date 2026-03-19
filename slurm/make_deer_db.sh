#!/bin/bash
python_interpreter='/home/geuba03p/miniconda3/envs/deer_project_2/bin/python'
data_folder='/projects/sciences/zoology/geurten_lab/files_extracted/new_file_sync/data/'
python_script="/home/geuba03p/PyProjects/headshake_project/DeerInfo.py"
engine_url="sqlite:///projects/sciences/zoology/geurten_lab/files_extracted/new_file_sync/deer_data.db"

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
