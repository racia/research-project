#!/bin/bash
#
# Job name
#SBATCH --job-name=feedback

#SBATCH --time=00:30:00              # Job time limit (30 minutes)
#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --gres=gpu:2                 # Request 2 GPUs
#SBATCH --cpus-per-task=2            # Number of CPU cores per task
#SBATCH --mem=32G                    # Total memory requested
#SBATCH --partition=dev_gpu_4

# Output and error logs
#SBATCH --output="feedback_out.txt"
#SBATCH --error="feedback_err.txt"

# Email notifications
#SBATCH --mail-user=""              # TODO: Add your email address
#SBATCH --mail-type=START,END,FAIL  # Send email when the job ends or fails

### JOB STEPS START HERE ###

if command -v module >/dev/null 2>&1; then
    echo "Module util is available. Loading miniconda and CUDA..."
    module load devel/miniconda/23.9.0-py3.9.15
    module load devel/cuda/11.8
else
    echo "Module util is not available. Using manually installed miniconda and CUDA..."
fi

# initialize shell to work with bash
source ~/.bashrc 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh

# Verify conda availability
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not available after loading the module."
    exit 1
else
    echo "Conda is available."
fi

#reinitialize conda
source ~/miniconda3/etc/profile.d/conda.sh

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

# Monitor GPU usage in background
(
    while true; do
        echo "== GPU Status: $(date) =="
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
        sleep 30
    done
) > gpu_monitor.log &
MONITOR_PID=$!#

# Run the Python script
SCRIPT="running_script.py"

# Set the environment variable to allow PyTorch to allocate more memory
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"


# declare array of config paths and names, e.g. "/path/to/config config_name"
# TODO: add config(s) to array
declare -a CONFIGS=(
  "$HOME/research-project/settings/feedback/config feedback_config"
)

for CONFIG in "${CONFIGS[@]}"
do
  CONFIG_PATH=$(echo $CONFIG | cut -d ' ' -f 1)
  CONFIG_NAME=$(echo $CONFIG | cut -d ' ' -f 2)
  echo "Running the script with config: CONFIG_PATH=$CONFIG_PATH, CONFIG_NAME=$CONFIG_NAME"
  python3 "$SCRIPT" --config-path $CONFIG_PATH --config-name $CONFIG_NAME hydra/job_logging=none
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