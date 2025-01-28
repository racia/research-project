# Reasoning Research Project

This repository serves to implement the research project on reasoning capabilities
of language models and the efforts to improve them through a series of prompting
and fine-tuning experiments.

The project is held as part of the research module.
More information
in [the main document](https://docs.google.com/document/d/1f44Xf9sQiklHzP1T34o62FgyKT4NHdeNGCIkNRJR7vY/edit?usp=sharing).

Authors: [@ivakhnenko](https://gitlab.cl.uni-heidelberg.de/ivakhnenko),
[@lingemann](https://gitlab.cl.uni-heidelberg.de/lingemann),
[@Motmem](https://gitlab.cl.uni-heidelberg.de/Motmem),
[@sari](https://gitlab.cl.uni-heidelberg.de/sari)

---

## Repository Structure

* [baseline](./baseline) — methods for running baseline and skyline
    * [config](baseline/config) - config files for running scripts

* [data](./data)
    * golden evaluation dataset
    * methods for data processing
    * methods to load and save data

* [prompts](./prompts) — prompt files

* [settings](./settings)
    * [feedback](./settings/feedback) — methods
    * [speculative-decoding](./settings/speculative-decoding) — methods

* [interpretability](./interpretability) — methods

* [prompting-experiments](./prompting-experiments) — scripts (prompts and settings combined)
    * [results](./prompting-experiments/results)

* [evaluation](./evaluation)
    * [interpretability](./evaluation/interpretability) — scripts (methods + data)
    * [testing](./evaluation/testing) — scripts (predicted data + golden data)

* [fine-tuning](./fine-tuning) — scripts (models + data)
    * [results](./fine-tuning/results)

* [tests](./tests) — tests for all the methods
    * [baseline](./tests/baseline)
    * [data](./tests/data)
    * [evaluation](./tests/evaluation)
    * [fine-tuning](./tests/fine-tuning)
    * [prompting-experiments](./tests/prompting-experiments)
    * [settings](./tests/settings)
        * [feedback](./tests/settings/feedback)
        * [speculative-decoding](./tests/settings/speculative-decoding)

---

## Getting started

### Set up the project

First, you need to get the data, the project repository,
and create an environment for the project.

**Important!** The structure of files that the scripts currently expect:

`root` (might be just a project folder)

* `tasks_1-20_v1-2` (bAbI tasks)
* `research-project`

The data repository on the cluster is located here: `/workspace/students/reasoning`.

#### Steps locally or on CLuster

0. [if CLuster is your destination] Connect to the cluster with `ssh {your_surname}@cluster.cl.uni-heidelberg.de`.


1. Make sure you are in the correct directory
    * if you are on the cluster, go to your root repository: `cd ~`,
    * otherwise to your project directory: `cd path/to/project/dir`


2. Install miniconda3:
    ```commandline
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    ```


3. Download the [bAbI task data](https://www.kaggle.com/datasets/roblexnana/the-babi-tasks-for-nlp-qa-system) from
   Kaggle
    * by importing kaggle:
    ```python
    import kagglehub
    
    # Download latest version
    path = kagglehub.dataset_download("roblexnana/the-babi-tasks-for-nlp-qa-system")
    
    print("Path to dataset files:", path)
    ```
    * by downloading a zip file and transferring the files to the cluster from your local machine with `rsync` or
      `sftp`.

        * You can run this example from another commandline tab **locally**:
      ```commandline
      sftp your_surname@cluster.cl.uni-heidelberg.de
      ```
      Commands for the cluster: `pwd`, `cd`, `ls`.

      Commands for your local machine: `lpwd`, `lcd`, `lls`.

      To move the files from your local machine to the cluster:
      ```commandline
      put local_path remote_path
      ```
      To move the files from the cluster to your local machine:
      ```commandline
      get remote_path local_path
      ```
      To move the directory, add `-r`, for example `put -r local_path remote_path`.


4. Clone the `research-project repository` with
    ```commandline
    git clone https://gitlab.cl.uni-heidelberg.de/sari/research-project.git
    ```

5. Create a conda environment for the project:
    ```commandline
    cd research-project
    source ~/miniconda3/etc/profile.d/conda.sh
    conda env create -f environment.yaml
    ```

   If you run into `CondaMemoryError: The conda process ran out of memory`, try a manual version
   (slower but seems to work):
    ```commandline
   conda create -n research-project python torchvision scikit-learn pytorch::pytorch torchaudio numpy transformers matplotlib black accelerate hydra-core -c conda-forge -c pytorch
    ```

6. Activate the environment: `conda activate research-project`.


7. Install other dependencies, if any.


8. Running models from the Hugging Face hub requires an access token,
   which you can obtain via the website on your https://huggingface.com profile.
   Save it as an environment variable in bash:
    ```commandline
    export HUGGINGFACE="<<your-token>>"
    ```

If you need to update the environment file with new dependencies, use this command

```commandline
conda env export --from-history > environment.yaml
```

Otherwise, automatic installation of packages will not work.

##### Alternative environment setup with pip3

To get started, log into the Heidelberg University Computational Linguistics cluster and in your home directory:

1. Create a virtual environment: `python3 -m venv venv`
2. Install all dependencies: `pip install -r requirements.txt`
3. Activate environment: `source ~/venv/bin/activate`

#### Steps on bwUniCluster

0. Connect to the bwUniCluster using ssh with the following command:
   `ssh hd_{uni_id}@bwunicluster.scc.kit.edu`.

1. Clone the GitHub repository. This can be done via HTTPS, so you do not need to setup an ssh token.
    ```commandline
    git clone https://gitlab.cl.uni-heidelberg.de/sari/research-project.git
    ```
   The repository will be created in your home directory.

2. In a new terminal, navigate to the local folder where the data is located. Then run the following command to copy the
   data to the remote home directory:
   `scp -r tasks_1-20_v1-2 hd_{uni_id}@bwunicluster.scc.kit.edu:.`

3. Activate the necessary modules:
    ```commandline
    module load devel/miniconda/23.9.0-py3.9.15
    module load devel/cuda/11.8
    ```

   You can check the available modules using `module avail`.

4. Create the conda environment by running the following commands.

- First, we need to change the solver: `conda config --set solver libmamba`
- Then, we can actually create the environment:
    ```commandline
    conda create -n research-project "python>=3.9" scikit-learn numpy transformers matplotlib black "hydra-core>1" "pytorch::pytorch>=2.0" torchvision torchaudio pytorch-cuda=11.8 "conda-forge::accelerate>=0.26.0" -c pytorch -c nvidia -c conda-forge
    ```

  In theory, this should also be possible using the `environment.yaml` file:
    ```commandline
    conda env create -f research-project/environment.yaml
    ```

5. Sometimes, the shell needs to be configured to use `conda activate`. If so, run `conda init bash`, and reload the
   shell, i.e. by closing the connection to the server and reconnecting again.

6. Activate the conda environment: `conda activate research-project`

7. To run the scripts with a Llama-3 model, you need to add a Huggingface access token:
    ```commandline
    export HF_TOKEN="<<your-token>>"
    ```

---

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

### Data

The data can be read and preprocessed using the datahandler. The preprocessing includes:

* Splitting the text files into samples. The ID of each sample is used as a key in the data dictionary.
* For each sample, split the sample in context lines, questions, answers, and supporting facts.
* For each line, remove newlines as well as trailing and leading whitespaces.

The preprocessed data is saved as a dictionary of the following format:

```
{sample_id:
    {"context:
        {line_number: line,
         line_number: line, ... }
     "question":
        {line_number: line,
         line_number: line, ... }
    "answer":
        {line_number: answer,
         line_number: answer, ... }
    "supporting_fact": [[line_number_first_answer, ...], [line_number_second_answer, ...], ...]
    }
}
```

Below is an example:

```
{0: 
    {
        'context': {
            1: 'Mary moved to the bathroom.', 
            2: 'John went to the hallway.', 
            4: 'Daniel went back to the hallway.', 
            5: 'Sandra moved to the garden.', 
            7: 'John moved to the office.', 
            8: 'Sandra journeyed to the bathroom.', 
            10: 'Mary moved to the hallway.', 
            11: 'Daniel travelled to the office.', 
            13: 'John went back to the garden.', 
            14: 'John moved to the bedroom.'
            }, 
        'question': {
            3: 'Where is Mary?', 
            6: 'Where is Daniel?', 
            9: 'Where is Daniel?', 
            12: 'Where is Daniel?', 
            15: 'Where is Sandra?'
            }, 
        'answer': {
            3: 'bathroom', 
            6: 'hallway', 
            9: 'hallway', 
            12: 'office', 
            15: 'bathroom'
            }, 
        'supporting_fact': [1, 4, 4, 11, 8]
    }
}
```

### Prompts

### Settings

#### Feedback

#### Speculative Decoding

### Interpretability

### Prompting Experiments

### Evaluation

### Fine-Tuning

### Tests
