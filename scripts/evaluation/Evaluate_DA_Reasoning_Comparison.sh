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

#SBATCH --mail-user=""              # TODO: Add your email address
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
ENV_NAME="research-project-4"
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
RUN_WITH_REAS="/workspace/students/reasoning/results/basic-baseline/test/da/average_run/joined_direct_answer_results_averaged.csv"
RUN_WITHOUT_REAS="/workspace/students/reasoning/results/basic-baseline/test/reasoning/average_run/joined_reasoning_results_averaged.csv"
OUT_DIR="/workspace/students/reasoning/results/analysis/basic-baseline/test/average_comparison/"

python3 evaluate_da_reasoning.py \
   --run_with_reas $RUN_WITH_REAS \
   --run_without_reas $RUN_WITHOUT_REAS \
   --out_dir $OUT_DIR

### BASELINE DA vs REASONING COMPARISON ###
echo "Comparing Direct Answer and Reasoning results for the Baseline setting..."
RUN_WITH_REAS="/workspace/students/reasoning/results/baseline/test/da/average_run/joined_direct_answer_results_averaged.csv"
RUN_WITHOUT_REAS="/workspace/students/reasoning/results/baseline/test/reasoning/average_run/joined_reasoning_results_averaged.csv"
OUT_DIR="/workspace/students/reasoning/results/analysis/baseline/test/average_comparison/"

python3 evaluate_da_reasoning.py \
   --run_with_reas $RUN_WITH_REAS \
   --run_without_reas $RUN_WITHOUT_REAS \
   --out_dir $OUT_DIR

### SKYLINE DA vs REASONING COMPARISON ###
echo "Comparing Direct Answer and Reasoning results for the Skyline setting..."
RUN_WITH_REAS="/workspace/students/reasoning/results/skyline/test/da/v1/joined_direct_answer_results.csv"
RUN_WITHOUT_REAS="/workspace/students/reasoning/results/skyline/test/reasoning/v1/joined_reasoning_results.csv"
OUT_DIR="/workspace/students/reasoning/results/analysis/skyline/test/v1_comparison/"

python3 evaluate_da_reasoning.py \
   --run_with_reas $RUN_WITH_REAS \
   --run_without_reas $RUN_WITHOUT_REAS \
   --out_dir $OUT_DIR

echo "Job completed successfully."
echo "Deactivating the environment: $ENV_NAME"
conda deactivate