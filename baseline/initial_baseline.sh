#!/bin/bash
#
#SBATCH --job-name=prompt_0
#SBATCH --output=prompt_0.txt
#SBATCH --gres=gpu:2
#SBATCH --mem=32000
#SBATCH --ntasks=1
#SBATCH --partition=students

# JOB STEPS
<<<<<<< HEAD
source ~/miniconda3/etc/profile.d/conda.sh
conda activate research
srun python3 script.py
conda activate
=======
bash
conda init
conda activate research
srun python3 script.py
conda deactivate
>>>>>>> c2433c8 (Modify details)
