#!/bin/bash
#
#SBATCH --time=00:30:00              # Job time limit (30 minutes)
#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --gres=gpu:2                 # Request 2 GPUs
#SBATCH --cpus-per-task=1            # Number of CPU cores per task


# Email notifications
#SBATCH --mail-user=""
#SBATCH --mail-type=END,FAIL  # Send email when the job ends or fails

# Output and error logs
#SBATCH --output="SD_out.txt"        # Standard output log
#SBATCH --error="SD_err.txt"         # Error log

# Job name
#SBATCH --job-name=SD

### JOB STEPS START HERE ###

# Verify conda availability
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not available after loading the module."
    exit 1
else
    echo "Conda is available."
fi

# initialize shell to work with bash
source ~/.bashrc

# Activate the conda environment
ENV_NAME="research-project"
echo "Activating conda environment: $ENV_NAME"
if ! conda activate "$ENV_NAME"; then
    echo "Error: Failed to activate conda environment '$ENV_NAME'."
    exit 1
else
    echo "Conda environment '$ENV_NAME' activated successfully."
fi

# Check if data directory exists
DATA_DIR="$HOME/tasks_1-20_v1-2"
echo "Checking for data directory: $DATA_DIR"
if [ -d "$DATA_DIR" ]; then
    echo "Data directory '$DATA_DIR' exists."
else
    echo "Error: Data directory '$DATA_DIR' does not exist."
    exit 1
fi

# Run the Python script
SCRIPT="run_SD.py"

# declare array of config paths and names, e.g. "/path/to/config config_name"
declare -a CONFIGS=(
  "$HOME/research-project/settings/SD/config SD_config"
)

for CONFIG in "${CONFIGS[@]}"
do
  CONFIG_PATH=$(echo $CONFIG | cut -d ' ' -f 1)
  CONFIG_NAME=$(echo $CONFIG | cut -d ' ' -f 2)
  echo "Running the script with config: CONFIG_PATH=$CONFIG_PATH, CONFIG_NAME=$CONFIG_NAME"
  srun python3 "$SCRIPT" --config-path $CONFIG_PATH --config-name $CONFIG_NAME
done

# Verify if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Python script '$SCRIPT' executed successfully."
else
    echo "Error: Python script '$SCRIPT' failed."
    exit 1
fi

echo "Job completed successfully."
