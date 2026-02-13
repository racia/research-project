#!/bin/bash
#
# Job name
#SBATCH --job-name=bbase

#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=2            # Number of CPU cores per task
#SBATCH --time=1:10:00              # Time limit hrs:min:sec
#SBATCH --mem=64G                    # Total memory requested
#SBATCH --partition=gpu_a100_il  # gpu_a100_short dev_gpu_h100

# Output and error logs
#SBATCH --output="basic_baseline_out.txt"        # TODO: adjust standard output log
#SBATCH --error="basic_baseline_err.txt"         # TODO: adjust error log

# Email notifications
#SBATCH --mail-user=""              # TODO: Add your email address
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email when the job ends or fails

### JOB STEPS START HERE ###
# fix working directory
cd ~/research-project || exit 1

if command -v module >/dev/null 2>&1; then
    echo "Module util is available. Loading python and CUDA..."
    module load devel/python/3.12.3-gnu-14.2
    module load devel/cuda/12.8
else
    echo "Module util is not available. Using manually installed python and CUDA..."
fi

# conda install nvidia::cuda==12.4.0
# conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# initialize shell to work with bash
source ~/.bashrc 2>/dev/null

# Activate the conda environment
#ENV_NAME="research-project-3"
#echo "Activating the project environment: $ENV_NAME"
#if ! conda activate $ENV_NAME; then
#    echo "Error: Failed to activate the project environment '$ENV_NAME'."
#    exit 1
#else
#    echo "The project environment '$ENV_NAME' activated successfully."
#fi
ENV_NAME=".env"
echo "Activating the project environment: $ENV_NAME"
if ! source $ENV_NAME/bin/activate; then
    echo "Error: Failed to activate the project environment '$ENV_NAME'."
    exit 1
else
    echo "The project environment '$ENV_NAME' activated successfully."
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
#declare -a CONFIGS=(
#  "$HOME/research-project/settings/baseline/config basic_baseline_test_da"
#)
#
#for CONFIG in "${CONFIGS[@]}"
#do
#  CONFIG_PATH=$(echo $CONFIG | cut -d ' ' -f 1)
#  CONFIG_NAME=$(echo $CONFIG | cut -d ' ' -f 2)
#  echo "Running the script for DA with config: CONFIG_PATH=$CONFIG_PATH, CONFIG_NAME=$CONFIG_NAME"
#  srun python3 "$SCRIPT" --config-path $CONFIG_PATH --config-name $CONFIG_NAME hydra/job_logging=none
#done
#
## Verify if the script executed successfully
#if [ $? -eq 0 ]; then
#    echo "Python script '$SCRIPT' executed successfully."
#else
#    echo "Error: Python script '$SCRIPT' failed."
#    exit 1
#fi

declare -a CONFIGS=(
  "$HOME/research-project/settings/baseline/config basic_baseline_test_reasoning"
)

for CONFIG in "${CONFIGS[@]}"
do
  CONFIG_PATH=$(echo $CONFIG | cut -d ' ' -f 1)
  CONFIG_NAME=$(echo $CONFIG | cut -d ' ' -f 2)
  echo "Running the script for Reasoning with config: CONFIG_PATH=$CONFIG_PATH, CONFIG_NAME=$CONFIG_NAME"
  srun python3 "$SCRIPT" --config-path $CONFIG_PATH --config-name $CONFIG_NAME hydra/job_logging=none
done

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