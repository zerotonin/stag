#!/bin/bash

# Directory containing .h5 files
H5_FILES_DIR="/projects/sciences/zoology/geurten_lab/deer_2024/files_extracted/sync_file_results/data"

# Initialize a variable to hold the job ID of the last submitted job
last_job_id=""

# Iterate through each .h5 file in the directory
for h5_file in "${H5_FILES_DIR}"/*.h5; do
    # Extract the basename of the h5 file for use in job name or other purposes
    h5_basename=$(basename "$h5_file")

    # Check if there is a last job ID to depend on
    if [ -n "$last_job_id" ]; then
        # Submit the job with a dependency on the last job's successful completion
        job_output=$(sbatch --dependency=afterok:$last_job_id <<EOF
#!/bin/bash
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gps_ana_$h5_basename  # Unique job name using the h5 file basename
#SBATCH --output=%x_%j.out  # Save output to a file based on the job name and job ID

~/miniconda3/envs/deer_project_2/bin/python /home/geuba03p/PyProjects/headshake_project/GPSAnalysis.py "/projects/sciences/zoology/geurten_lab/deer_2024/deer_data.db" "$h5_file"
EOF
)
    else
        # Submit the first job without any dependency
        job_output=$(sbatch <<EOF
#!/bin/bash
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gps_ana_$h5_basename  # Unique job name using the h5 file basename
#SBATCH --output=%x_%j.out  # Save output to a file based on the job name and job ID

~/miniconda3/envs/deer_project_2/bin/python /home/geuba03p/PyProjects/headshake_project/GPSAnalysis.py "/projects/sciences/zoology/geurten_lab/deer_2024/deer_data.db" "$h5_file"
EOF
)
    fi

    # Extract the job ID from the sbatch output and update the last_job_id variable
    last_job_id=$(echo "$job_output" | awk '{print $4}')
done
