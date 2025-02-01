# The Settings

At some point, here should be some detailed information about the different settings that can be run.

I imagine, we would include some basic instruction on how to run a setting and some more detailed information about the
settings and their structure.

## The information below is not up-to-date and will be revised at some point.

### Baseline

#### Running the baseline on the CLuster and bwUniCluster

1. Change directory to the subproject folder: `cd ~research-project/baseline`.


2. Create your `yaml` files with configs in `research-project/config`.
   If you use defaults, you can only specify a detail or two,
   and the rest will be taken from the default config file.


3. Create your bash script from the default `baseline.sh`. You can specify:
    * the name of the job
    * name of the output file ()
    * number of CPU's
    * memory to allocate (if too big, longer waiting time)
    * patrition (default — `students`)
    * most importantly, the list of config files you want to run the script with:
      ``` 
      CONFIGS="prompt_0_shot prompt_1_shot"
      ```
      The script runs with each config separately.


4. Submit a batch job: `sbatch name_of_script.sh`.


5. Check the status with `squeue`.

#### Running the baseline locally

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

### Settings

As the settings load two Llama models, it not possible to run these settings on CLuster or locally.

#### Running the settings on the bwUniCluster

Make sure to activate the conda module and the CUDA module. Furthermore, check that your pytorch version supports GPU
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