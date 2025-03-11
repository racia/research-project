# The Settings

This project has four different settings. A baseline and a skyline setting, which are used to compare the performance of
our approaches. The other settings, which are the main focus of this work, are the feedback setting and the speculative
decoding setting.

## Baseline

The baseline consists of the Llama-3-8b model.

## Skyline

The skyline consists of the Llama-3-70b model.

## Feedback

In the Feedback setting, the student starts by generating an initial chain-of-thought and an answer to the question.

This CoT and the answer are then given to the teacher. The teacher now reads the generated CoT and provides some
feedback for the student. This feedback can be either that the student needs to further refine its CoT or that the CoT
is plausible as it is.

In the case that the teacher found that some further refinement is necessary, the student receives its own CoT again as
well as the feedback that was provided by the teacher. Based on this, it is then prompted to refine its CoT.

This refined CoT is then again given to the teacher in order to get some feedback on it. This process of the student
refining its CoT based on some feedback and the teacher given feedback on this new CoT is repeated until the teacher
accepts the CoT.

## Speculative Decoding

In the Speculative Decoding setting, the student starts by generating a whole initial chain-of-thought and an answer to
the question.

This CoT is then given to the teacher. The teacher goes through the chain of thought TOKEN BY TOKEN? REASONING STEP BY
REASONING STEP? CHUNK BY CHUNK?. As soon as the teacher deems a x incorrect, it provides a correction for the faulty x.

The students correct part of the CoT as well as the intervention x by the teacher is then given back to the student.
Based on the correction, it is then prompted to continue its CoT, starting after the intervention x.

## Running a setting on the CLuster and bwUniCluster

Due to the size of the big Llama model, it is only possible to run the baseline on CLuster. On the bwUniCluster, all of
our settings can be run.

1. Change directory to the subproject folder: `cd ~research-project/baseline`.


2. Create your `yaml` files with configs in `research-project/config`.
   If you use defaults, you can only specify a detail or two,
   and the rest will be taken from the default config file.


3. Use the default `setting_{server}.sh`. You should specify:
    * the name of the job
    * name of the output file
    * number of CPU's
    * memory to allocate (if too big, longer waiting time)
    * partition
    * your email adress to receive notifications
    * the list of config files you want to run the script with:
       ``` 
       "$HOME/research-project/settings/baseline/config baseline_config_bwUniCluster"
       ```
      The script runs with each config separately.


4. Submit a batch job: `sbatch setting_{server}.sh`.


5. Check the status with `squeue`.

## Running a setting locally

Due to the size of the Llama models, it is only possible to run the baseline locally.

1. Change directory to the subproject folder: `cd ~research-project/baseline`.


2. Create your `yaml` files with configs in `research-project/config`.
   If you use defaults, you can only specify a detail or two,
   and the rest will be taken from the default config file.
   ! You need to adjust the result directory for each run! Otherwise your old results will be overwritten!


3. Run the script with a config file, for example `baseline_config`:
    ```commandline
    python3 baseline_script.py --config-path "config/baseline_config" --config-name "baseline_config"
    ```
   If no config specified, the script will run with the default `baseline_config`
   (will through an error of not finding `/workspace/students/reasoning`).

### Notes

#### Running the settings on the bwUniCluster

Make sure to activate the conda module and the CUDA module.

```commandline
module load devel/miniconda/23.9.0-py3.9.15
module load devel/cuda/11.8
``` 

Furthermore, check that your pytorch version supports GPU
usage.
The code will stop executing if no GPU is available to use.

We are now using two GPUs, `cuda:0` and `cuda:1`.
The student model is loaded on `cuda:0`, the teacher on `cuda:1`.
This is done because we otherwise do not have enough GPU memory to do any computations.
If we now want to generate something, we need the input to the model to be on the same device as the model that should
generate.
So if we want to generate the initial CoT of the student, the input to the model (in this case the encoded prompt) needs
to be moved to `cuda:0` as well. This can simply be done by adding `.to("cuda:0")` to the tensor.
We can then happily do some stuff with this, until we want to get the teacher to do something. Then we need to again
move the input to the teacher (e.g. the prompt to provide feedback + the CoT of the student) to `cuda:1`, where the
teacher is located.

Therefore you need to make sure that the batch script specifies to use two GPUs.
This also requires that the data is moved correctly in the code as explained above. If you want to change something,
make sure that the necessary data is on the same GPU as the device that should use it!