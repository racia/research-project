#!/bin/bash
#
#SBATCH --job-name=prompt_11_
#SBATCH --output=results/prompt_11_.txt
#SBATCH --gres=gpu:2
#SBATCH --mem=32000
#SBATCH --ntasks=1
#SBATCH --partition=students

# JOB STEPS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate research
srun python3 script.py
conda activate
