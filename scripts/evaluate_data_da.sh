#!/bin/bash
#SBATCH --job-name=eval_data
#SBATCH --output=eval_data_%j.out
#SBATCH --error=eval_data_%j.err
#SBATCH --time=06:00:00 # should be enough for 20 samples per task
#SBATCH --cpus-per-task=4
# SBATCH --partition=cpu
#SBATCH --mail-user=""
#SBATCH --mail-type=BEGIN,END,FAIL

experiment="da"
setting=$1
samples_per_task=$2
create_heatmaps=$3

if [ $experiment = "reasoning" ]; then
    full_mode="reasoning"
else
    full_mode="direct_answer"
fi

if [ "$setting" = "sd" ]; then
    setting_adj="SD"
else
    setting_adj="$setting"
fi

results_path="/pfs/work9/workspace/scratch/hd_mr338-research-results-2/${setting_adj}/test/${experiment}/v1/all_tasks_joined/joined_${full_mode}_results.csv"

save_path="/pfs/work9/workspace/scratch/hd_mr338-research-results-2/analysis/${setting_adj}/${experiment}"
# save_path="results/${setting}/${mode}"

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

echo "Evaluating data for setting: $setting, task: $task, samples per task: $samples_per_task, create heatmaps: $create_heatmaps"

srun python3 evaluate_data.py \
    --results_path $results_path \
    --save_path $save_path \
    --samples_per_task $samples_per_task \
    --experiment $full_mode \
    --setting $setting \
    ${create_heatmaps:+--create_heatmaps}