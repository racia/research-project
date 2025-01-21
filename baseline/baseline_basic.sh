#!/bin/bash
#
#SBATCH --job-name=baseline
#SBATCH --gres=gpu:2
#SBATCH --mem=32000
#SBATCH --ntasks=1
#SBATCH --partition=students

# Email notifications
#SBATCH --mail-user=""
#SBATCH --mail-type=END,FAIL  # Send email when the job ends or fails

# JOB STEPS
# shellcheck source=/dev/null
source ~/miniconda3/etc/profile.d/conda.sh
conda activate research-project

# run the script a required number of times with various settings
# example of a list: "baseline_config_1 baseline_config_2 baseline_config_2"
CONFIGS="prompt_0_shot prompt_1_shot prompt_2_shot prompt_3_shot prompt_4_shot prompt_5_shot"

for CONFIG in $CONFIGS
do
    echo "Running the script with config: $CONFIG"
    srun python3 baseline_script.py --config "$CONFIG"
done

conda activate
