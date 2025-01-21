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
#SBATCH --output="baseline_out.txt"        # Standard output log
#SBATCH --error="baseline_err.txt"         # Error log

# Job name
#SBATCH --job-name=baseline

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
    echo "Error: Data directory '$DATA_DIR' does not exist in your home directory."
    exit 1
fi

# Run the Python script
SCRIPT="baseline_script.py"

# Set the environment variable to allow PyTorch to allocate more memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# declare array of config paths and names, e.g. "/path/to/config config_name"
declare -a CONFIGS=(
  "$HOME/research-project/baseline/config baseline_config_bwUniCluster"
)

# OUTPUT_DIR will be re-declared in the script for each config,
# here it is just to be on the safe side
OUTPUT_DIR="$HOME/research-project/output"
LOCAL_MACHINE_USER="bohdana.ivakhnenko"
# there should be no VPN connection, otherwise the following command will yield two IP addresses
LOCAL_MACHINE_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | cut -d\  -f2)
LOCAL_DESTINATION_PATH="/Users/bohdana.ivakhnenko/PycharmProjects/research-project/baseline"

for CONFIG in "${CONFIGS[@]}"
do
  CONFIG_PATH=$(echo $CONFIG | cut -d ' ' -f 1)
  CONFIG_NAME=$(echo $CONFIG | cut -d ' ' -f 2)
  echo "Running the script with config: CONFIG_PATH=$CONFIG_PATH, CONFIG_NAME=$CONFIG_NAME"
  srun python3 "$SCRIPT" --config-path $CONFIG_PATH --config-name $CONFIG_NAME hydra/job_logging=none

  echo "Copying output folder of CONFIG_NAME=$CONFIG_NAME to local machine..."
  # OUTPUT_DIR is declared in the script
  scp -r "$OUTPUT_DIR" "$LOCAL_MACHINE_USER@$LOCAL_MACHINE_IP:$LOCAL_DESTINATION_PATH"

  if [ $? -eq 0 ]; then
      echo "Output folder copied successfully."
  else
      echo "Error: Failed to copy the output folder."
      exit 1
  fi
done

# Verify if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Python script '$SCRIPT' executed successfully."
else
    echo "Error: Python script '$SCRIPT' failed."
    exit 1
fi

echo "Job completed successfully."

COLUMNS="JobID,JobName,MaxRSS,NTasks,AllocCPUS,AllocGRES,AveDiskRead,AveDiskWrite,Elapsed,State"
sacct -l -j $SLURM_JOB_ID --format=$COLUMNS

echo "Deactivating conda environment: $ENV_NAME"
conda deactivate