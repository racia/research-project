#!/bin/bash
#SBATCH --job-name=evale_data
#SBATCH --output=eval_data_%j.out
#SBATCH --error=eval_data_%j.err
#SBATCH --time=02:00:00 # should be enough for 20 samples per task
#SBATCH --cpus-per-task=4
# SBATCH --partition=dev_cpu_il
#SBATCH --mail-user=sari@cl.uni-heidelberg.de
#SBATCH --mail-type=BEGIN,END,FAIL

setting=$1
task=$2
samples_per_task=$3
create_heatmaps=$4

results_path="/pfs/work9/workspace/scratch/hd_mr338-research-results-2/${setting}/test/${task}/v1/all_tasks_joined/joined_reasoning_results.csv"
save_path="results/${setting}/${task}"

source .env/bin/activate
echo "Evaluating data for setting: $setting, task: $task, samples per task: $samples_per_task, create heatmaps: $create_heatmaps"

if [ "$create_heatmaps" = "true" ]; then
    srun python3.9 evaluate_data.py --results_path $results_path --save_path $save_path --samples_per_task $samples_per_task --create-heatmaps 
else
    srun python3.9 evaluate_data.py --results_path $results_path --save_path $save_path --samples_per_task $samples_per_task
fi

