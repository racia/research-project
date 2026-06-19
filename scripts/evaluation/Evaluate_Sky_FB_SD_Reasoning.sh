#!/bin/bash
#
# Job name
#SBATCH --job-name=single_versions

#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --cpus-per-task=2 #4            # Number of CPU cores per task
#SBATCH --mem=16GB                    # Total memory requested
#SBATCH --partition=students
# SBATCH --time=00:30:00              # Job time limit (30 minutes)
# Output and error logs
#SBATCH --output="eval_sky_fb_sd_reasoning_%j.log"

#SBATCH --mail-user=""              # TODO: Add your email address
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

# Toggle args here
VERBOSE=true #true
HEATMAPS=false #true
# Set to "claude" or "llama" to select a silver-reasoning corpus;
# leave empty to use the default flat directory (legacy behaviour).
REASONING_SOURCE="llama"  # "claude" | "llama" | ""

### SKYLINE REASONING ###
#echo "Evaluating Skyline Reasoning results for version v1..."
#RES_PATH="/workspace/students/reasoning/results/skyline/test/reasoning/v1/all_tasks_joined/joined_reasoning_results.csv"
##RES_PATH="/workspace/students/reasoning/results/skyline/test/reasoning/v1/all_tasks_joined/joined_direct_answer_results.csv"
#SAVE_PATH="/workspace/students/reasoning/results/analysis/skyline/reasoning/v1/"
##SAVE_PATH="results/skyline/reasoning"
#SETTING="skyline" #"skyline"
#EXPERIMENT="reasoning" #"reasoning" "direct_answer"
## TODO: turn dict into a mapping of setting to filtering conditions
##FILTERING_CONDITIONS='{"skyline": {"model": "gpt-3.5-turbo", "reasoning_type": "none"}, "chain_of_thought": {"model": "gpt-3.5-turbo", "reasoning_type": "chain_of_thought"}, "scratchpad": {"model": "gpt-3.5-turbo", "reasoning_type": "scratchpad"}}'
#SAMPLES_PER_TASK=100 #100
#MAX_TOKENS=... # TODO: check what average max tokens value was
## v1 - task 1-2, 6-9, 10-19: 150, tasks 3-5: 100, task 20: 12 # TODO: rerun task 20 with more tokens
#
#ARGS=(
#  --results_path "$RES_PATH"
#  --save_path "$SAVE_PATH"
#  --setting "$SETTING"
#  --experiment "$EXPERIMENT"
#  --samples_per_task "$SAMPLES_PER_TASK"
#  --max_tokens "$MAX_TOKENS"
#)
#
#[ "$VERBOSE" = true ] && ARGS+=(--verbose)
#[ "$HEATMAPS" = true ] && ARGS+=(--create_heatmaps)
#[ -n "$REASONING_SOURCE" ] && ARGS+=(--reasoning_source "$REASONING_SOURCE")
#
#SCRIPT="evaluate_data.py"
#echo "Running script ${SCRIPT} with the following arguments: ${ARGS[*]}"
#srun python3 "$SCRIPT" "${ARGS[@]}"

### FEEDBACK REASONING ###  # v1 is corrupted, so we only run v2
echo "Evaluating Feedback Reasoning results for version v2..."
RES_PATH="/workspace/students/reasoning/results/feedback/test/reasoning/v2/all_tasks_joined/joined_reasoning_results.csv"
#RES_PATH="/workspace/students/reasoning/results/feedback/test/reasoning/v2/all_tasks_joined/joined_direct_answer_results.csv"
SAVE_PATH="/workspace/students/reasoning/results/analysis/feedback/reasoning/v2/"
#SAVE_PATH="results/feedback/reasoning"
SETTING="feedback" #"feedback"
EXPERIMENT="reasoning" #"reasoning" "direct_answer"
# TODO: turn dict into a mapping of setting to filtering conditions
#FILTERING_CONDITIONS='{"feedback": {"model": "gpt-3.5-turbo", "reasoning_type": "none"}, "chain_of_thought": {"model": "gpt-3.5-turbo", "reasoning_type": "chain_of_thought"}, "scratchpad": {"model": "gpt-3.5-turbo", "reasoning_type": "scratchpad"}}'
SAMPLES_PER_TASK=100 #100
MAX_TOKENS=250 # TODO: check what average max tokens value was
# v2 teacher - task 1: 250, tasks 2-3: 250-300, tasks 4-5, 7, 10-20: 300, task 6: 200
# v2 student - task 1: 250, tasks 2-3: 250-300, tasks 4-5, 7, 10-20: 300, task 6: 200

ARGS=(
  --results_path "$RES_PATH"
  --save_path "$SAVE_PATH"
  --setting "$SETTING"
  --experiment "$EXPERIMENT"
  --samples_per_task "$SAMPLES_PER_TASK"
  --max_tokens "$MAX_TOKENS"
)

[ "$VERBOSE" = true ] && ARGS+=(--verbose)
[ "$HEATMAPS" = true ] && ARGS+=(--create_heatmaps)
[ -n "$REASONING_SOURCE" ] && ARGS+=(--reasoning_source "$REASONING_SOURCE")

SCRIPT="evaluate_data.py"
echo "Running script ${SCRIPT} with the following arguments: ${ARGS[*]}"
srun python3 "$SCRIPT" "${ARGS[@]}"

### SD REASONING ###
echo "Evaluating SD Reasoning results for version v1..."
RES_PATH="/workspace/students/reasoning/results/SD/test/reasoning/v1/all_tasks_joined/joined_reasoning_results.csv"
#RES_PATH="/workspace/students/reasoning/results/SD/test/reasoning/v1/all_tasks_joined/joined_direct_answer_results.csv"
SAVE_PATH="/workspace/students/reasoning/results/analysis/SD/reasoning/v1/"
#SAVE_PATH="results/SD/reasoning"
SETTING="SD" #"SD"
EXPERIMENT="reasoning" #"reasoning" "direct_answer"
# TODO: turn dict into a mapping of setting to filtering conditions
#FILTERING_CONDITIONS='{"SD": {"model": "gpt-3.5-turbo", "reasoning_type": "none"}, "chain_of_thought": {"model": "gpt-3.5-turbo", "reasoning_type": "chain_of_thought"}, "scratchpad": {"model": "gpt-3.5-turbo", "reasoning_type": "scratchpad"}}'
SAMPLES_PER_TASK=100 #100
MAX_TOKENS=250 # TODO: check what average max tokens value was
# v1 teacher - task 1-4, 6-9, 10-19: 150, task 8, 20: 12 or 150
# v1 student - task 1-4, 6-9, 10-19: 250, task 8, 20: 200 or 250

ARGS=(
  --results_path "$RES_PATH"
  --save_path "$SAVE_PATH"
  --setting "$SETTING"
  --experiment "$EXPERIMENT"
  --samples_per_task "$SAMPLES_PER_TASK"
  --max_tokens "$MAX_TOKENS"
)

[ "$VERBOSE" = true ] && ARGS+=(--verbose)
[ "$HEATMAPS" = true ] && ARGS+=(--create_heatmaps)
[ -n "$REASONING_SOURCE" ] && ARGS+=(--reasoning_source "$REASONING_SOURCE")

SCRIPT="evaluate_data.py"
echo "Running script ${SCRIPT} with the following arguments: ${ARGS[*]}"
srun python3 "$SCRIPT" "${ARGS[@]}"

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