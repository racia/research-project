#!/bin/bash
#
#SBATCH --job-name=Init_Bl
#SBATCH --output=Init_Bl.txt
#SBATCH --gres=gpu:2
#SBATCH --mem=32000
#SBATCH --ntasks=1
#SBATCH --partition=students

# JOB STEPS
source ~/venv/bin/activate
srun python3 script.py

