#!/bin/bash
#
# Job name
#SBATCH --job-name=master_results

#SBATCH --time=6:00:00              # Job time limit (30 minutes)
#SBATCH --ntasks=1                   # Total number of tasks

# Output and error logs
#SBATCH --output="master_res_out.txt"
#SBATCH --error="master_res_err.txt"

# Email notifications
#SBATCH --mail-user=""
#SBATCH --mail-type=START,END,FAIL  # Send email when the job ends or fails

### JOB STEPS START HERE ###
# fix working directory
cd ~/research-project || exit 1

if command -v module >/dev/null 2>&1; then
    echo "Module util is available. Loading python"
    module load devel/python/3.12.3-gnu-14.2
else
    echo "Module util is not available. Using manually installed python..."
fi

# initialize shell to work with bash
source ~/.bashrc 2>/dev/null

# Activate the conda environment
ENV_NAME=".env"
echo "Activating the project environment: $ENV_NAME"
if ! source $ENV_NAME/bin/activate; then
    echo "Error: Failed to activate the project environment '$ENV_NAME'."
    exit 1
else
    echo "The project environment '$ENV_NAME' activated successfully."
fi

# Run the Python script
SCRIPT="scripts/create_master_result_file.py"

python3 "$SCRIPT"

# Verify if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Python script '$SCRIPT' executed successfully."
else
    echo "Error: Python script '$SCRIPT' failed."
    exit 1
fi

echo "Job completed successfully."
echo "Deactivating the environment: $ENV_NAME"
deactivate