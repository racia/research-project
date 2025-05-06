# Running Experiments

## Direct Answer

We'll run it for:

1. basic baseline
2. advanced baseline
3. skyline

## Reasoning

We'll run it for:

1. basic baseline
2. advanced baseline
3. skyline
4. feedback
5. speculative decoding

### Advanced Baseline

Date: 06.05.2025

Instruction:

0. log into the cluster
1. make sure that all of your current work there is saved and commited to your branch, so you are not adding any code on
   top of the current version
2. `git fetch`
3. `git checkout baseline-run`
4. `git pull`
5. run your sbatch script with all the possible partitions `sbatch --partition={partition_name} Baseline_{your_name}.sh`

   > NB! Possible GPU partitions are:
   > 1. gpu_h100
   > 2. gpu_mi300
   > 3. gpu_a100_il
   > 4. gpu_h100_il

   You might also want to check the available partitions: `sinfo_t_idle`

6. check job allocation with `squeue --start`
7. once a job is started, cancel all other jobs with `scancel {job_id}`