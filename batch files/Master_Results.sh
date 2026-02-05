#!/bin/bash
#
# Job name
#SBATCH --job-name=master_results

#SBATCH --time=6:00:00              # Job time limit (30 minutes)
#SBATCH --ntasks=1                   # Total number of tasks

# Output and error logs
#SBATCH --output="master_res_out.txt"
#SBATCH --error="master_res_err.txt"

# Email notifications
#SBATCH --mail-user=""
#SBATCH --mail-type=START,END,FAIL  # Send email when the job ends or fails

### JOB STEPS START HERE ###

# initialize shell to work with bash
source ~/.bashrc 2>/dev/null

# run bash commands
python3 scripts/create_master_result_file.py