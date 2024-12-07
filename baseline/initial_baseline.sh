#!/bin/bash
#
#SBATCH --job-name=prompt_0_
#SBATCH --output=results/prompt_0_.txt
#SBATCH --gres=gpu:2
#SBATCH --mem=32000
#SBATCH --ntasks=1
#SBATCH --partition=students
# JOB STEPS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate research
srun python3 baseline_script.py
conda activate
