#!/bin/bash
#
# Job name
#SBATCH --job-name=zip

#SBATCH --time=6:00:00              # Job time limit (30 minutes)
#SBATCH --ntasks=1                   # Total number of tasks

# Output and error logs
#SBATCH --output="zip_out.txt"
#SBATCH --error="zip_err.txt"

# Email notifications
#SBATCH --mail-user=""
#SBATCH --mail-type=START,END,FAIL  # Send email when the job ends or fails

### JOB STEPS START HERE ###

# initialize shell to work with bash
source ~/.bashrc 2>/dev/null

# run bash commands
zip -r /pfs/work9/workspace/scratch/hd_mr338-research-results-2/results_baseline.zip /pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline
zip -r /pfs/work9/workspace/scratch/hd_mr338-research-results-2/results_feedback.zip /pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback
zip -r /pfs/work9/workspace/scratch/hd_mr338-research-results-2/results_SD.zip /pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD
zip -r /pfs/work9/workspace/scratch/hd_mr338-research-results-2/results_skyline.zip /pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline