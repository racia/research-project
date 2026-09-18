#!/bin/bash
#
# Job name
#SBATCH --job-name=da_reas

#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --cpus-per-task=2 #4            # Number of CPU cores per task
#SBATCH --mem=16GB                    # Total memory requested
#SBATCH --partition=students
# SBATCH --time=01:00:00              # Job time limit (30 minutes)
# Output and error logs
#SBATCH --output="eval_da_reas_%j.log"

#SBATCH --mail-user="ivakhnenko@cl.uni-heidelberg.de"              # TODO: Add your email address
#SBATCH --mail-type=ALL  # Send email when the job ends or fails

### JOB STEPS START HERE ###
# fix working directory
cd ~/research-project || exit 1

#if command -v module >/dev/null 2>&1; then
#    echo "Module util is available. Loading python"
#    module load devel/python/3.12.3-gnu-14.2
#else
#    echo "Module util is not available. Using manually installed python..."
#fi

# initialize shell to work with bash
source ~/.bashrc 2>/dev/null

# Activate the conda environment
ENV_NAME="research-project-3"
conda activate $ENV_NAME
#ENV_NAME=".env"
#echo "Activating the project environment: $ENV_NAME"
#if ! source $ENV_NAME/bin/activate; then
#   echo "Error: Failed to activate the project environment '$ENV_NAME'."
#   exit 1
#else
#   echo "The project environment '$ENV_NAME' activated successfully."
#fi

### BASIC BASELINE DA vs REASONING COMPARISON ###
echo "Comparing Direct Answer and Reasoning results for the Basic Baseline setting..."
RUN_WITH_REAS="/workspace/students/reasoning/results/analysis/basic-baseline/reasoning/average_run/silver_llama/max_tokens_300/eval/"
RUN_DA="/workspace/students/reasoning/results/analysis/basic-baseline/da/average_run/eval/"
OUT_DIR="/workspace/students/reasoning/results/analysis/basic-baseline/average_comparison/"

python3 evaluate_da_reasoning.py \
   --run_with_reas $RUN_WITH_REAS \
   --run_da $RUN_DA \
   --out_dir $OUT_DIR

### BASELINE DA vs REASONING COMPARISON ###
echo "Comparing Direct Answer and Reasoning results for the Baseline setting..."
RUN_WITH_REAS="/workspace/students/reasoning/results/analysis/baseline/reasoning/average_run/silver_llama/max_tokens_300/eval/"
RUN_DA="/workspace/students/reasoning/results/analysis/baseline/da/average_run/eval/"
OUT_DIR="/workspace/students/reasoning/results/analysis/baseline/average_comparison/"

python3 evaluate_da_reasoning.py \
   --run_with_reas $RUN_WITH_REAS \
   --run_da $RUN_DA \
   --out_dir $OUT_DIR

### SKYLINE DA vs REASONING COMPARISON ###
echo "Comparing Direct Answer and Reasoning results for the Skyline setting..."
RUN_WITH_REAS="/workspace/students/reasoning/results/analysis/skyline/reasoning/v1/eval/silver_llama/max_tokens_150/"
RUN_DA="/workspace/students/reasoning/results/analysis/skyline/da/average_run/eval/"
OUT_DIR="/workspace/students/reasoning/results/analysis/skyline/ave_v1_comparison/"

python3 evaluate_da_reasoning.py \
   --run_with_reas $RUN_WITH_REAS \
   --run_da $RUN_DA \
   --out_dir $OUT_DIR

echo "Job completed successfully."
echo "Deactivating the environment: $ENV_NAME"
conda deactivate