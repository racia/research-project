#!/bin/bash
#
# Job name
#SBATCH --job-name=evaluate

#SBATCH --time=01:00:00              # Job time limit (30 minutes)
#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --cpus-per-task=2            # Number of CPU cores per task
#SBATCH --mem=1GB                    # Total memory requested
#SBATCH --partition=cpu

# Output and error logs
#SBATCH --output="eval_out.txt"        # TODO: adjust standard output log
#SBATCH --error="eval_err.txt"         # TODO: adjust error log

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

# Toggle args here
VERBOSE=true
HEATMAPS=false
RES_PATH="path/to/results"
SAVE_PATH="path/to/save"
SAMPLES_PER_TASK=100

ARGS=(
  --results_path "$RES_PATH"
  --save_path "$SAVE_PATH"
  --samples_per_task "$SAMPLES_PER_TASK"
)

[ "$VERBOSE" = true ] && ARGS+=(--verbose)
[ "$HEATMAPS" = true ] && ARGS+=(--create_heatmaps)

python "$SCRIPT" "${ARGS[@]}"


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