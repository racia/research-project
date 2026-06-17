#!/bin/bash
#
# Job name
#SBATCH --job-name=eval_sl

#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --cpus-per-task=2 #4            # Number of CPU cores per task
#SBATCH --mem=16GB                    # Total memory requested
# SBATCH --partition=students
#SBATCH --time=01:00:00              # Job time limit (30 minutes)
# Output and error logs
#SBATCH --output="eval_sl-da.out"        # TODO: adjust standard output log
# SBATCH --error="eval_sl-da.err"         # TODO: adjust error log

#SBATCH --mail-user="sari@cl.uni-heidelberg.de"              # TODO: Add your email address
#SBATCH --mail-type=ALL  # Send email when the job ends or fails

### JOB STEPS START HERE ###
# fix working directory
# cd ~/research-project || exit 1

#if command -v module >/dev/null 2>&1; then
#    echo "Module util is available. Loading python"
#    module load devel/python/3.12.3-gnu-14.2
#else
#    echo "Module util is not available. Using manually installed python..."
#fi

# initialize shell to work with bash
source ~/.bashrc 2>/dev/null

# Activate the conda environment
# ENV_NAME="research-project-4"
ENV_NAME=".env"
# conda activate $ENV_NAME
echo "Activating the project environment: $ENV_NAME"
if ! source $ENV_NAME/bin/activate; then
   echo "Error: Failed to activate the project environment '$ENV_NAME'."
   exit 1
else
   echo "The project environment '$ENV_NAME' activated successfully."
fi

# Toggle args here
VERBOSE=false #true
HEATMAPS=false #true
# Set to "claude" or "llama" to select a silver-reasoning corpus;
# leave empty to use the default flat directory (legacy behaviour).
REASONING_SOURCE=""  # "claude" | "llama" | ""
# RES_PATH="/workspace/students/reasoning/results/basic-skyline/test/reasoning/v1/all_tasks_joined/joined_reasoning_results.csv"
RES_PATH="/workspace/students/reasoning/results/skyline/test/da/v1/all_tasks_joined/joined_direct_answer_results.csv"
# SAVE_PATH="/workspace/students/reasoning/results/basic-skyline/test/reasoning/v1/all_tasks_joined/"
SAVE_PATH="results/skyline/da"
SETTING="skyline" #"skyline"
EXPERIMENT="direct_answer" #"da"
# TODO: turn dict into a mapping of setting to filtering conditions
#FILTERING_CONDITIONS='{"skyline": {"model": "gpt-3.5-turbo", "reasoning_type": "none"}, "chain_of_thought": {"model": "gpt-3.5-turbo", "reasoning_type": "chain_of_thought"}, "scratchpad": {"model": "gpt-3.5-turbo", "reasoning_type": "scratchpad"}}'
SAMPLES_PER_TASK=20 #100

ARGS=(
  --results_path "$RES_PATH"
  --save_path "$SAVE_PATH"
  --setting "$SETTING"
  --experiment "$EXPERIMENT"
  --samples_per_task "$SAMPLES_PER_TASK"
)

[ "$VERBOSE" = true ] && ARGS+=(--verbose)
[ "$HEATMAPS" = true ] && ARGS+=(--create_heatmaps)
[ -n "$REASONING_SOURCE" ] && ARGS+=(--reasoning_source "$REASONING_SOURCE")

SCRIPT="evaluate_data.py"
echo "Running script ${SCRIPT} with the following arguments: ${ARGS[*]}"
srun python3.12 "$SCRIPT" "${ARGS[@]}"


# Verify if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Python script '$SCRIPT' executed successfully."
else
    echo "Error: Python script '$SCRIPT' failed."
    exit 1
fi

echo "Job completed successfully."
echo "Deactivating the environment: $ENV_NAME"
conda deactivate