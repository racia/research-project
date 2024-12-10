#!/bin/bash
#
#SBATCH --job-name=prompt_1
#SBATCH --output=results/prompt_1.txt
#SBATCH --gres=gpu:2
#SBATCH --mem=32000
#SBATCH --ntasks=1
#SBATCH --partition=students
# JOB STEPS
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate research
source ../venv/bin/activate
srun python3 baseline_script.py
#conda activate
