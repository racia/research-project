#!/bin/bash
#
#SBATCH --job-name=prompt_1
#SBATCH --output=prompt_1.txt
#SBATCH --gres=gpu:2
#SBATCH --mem=32000
#SBATCH --ntasks=1
#SBATCH --partition=students

# JOB STEPS
bash
conda init
conda activate research
srun python3 script.py
conda deactivate
