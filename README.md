# Research Project

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

## Docs

### Getting started

First, you need to get the data, the project repository, 
and create an environment for the project.

**Important!** The structure of files that the scripts currently expect:

`root` (might be just a project folder)
* `tasks_1-20_v1-2` (bAbI tasks)
* `research-project`

The data repository on the cluster is located here: `/workspace/students/reasoning`.

#### Setting up the project

0. [optional] Connect to the cluster with `ssh your_surname@cluster.cl.uni-heidelberg.de`.


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


3. Download the [bAbI task data](https://www.kaggle.com/datasets/roblexnana/the-babi-tasks-for-nlp-qa-system) from Kaggle
    * by importing kaggle:
    ```python
    import kagglehub
    
    # Download latest version
    path = kagglehub.dataset_download("roblexnana/the-babi-tasks-for-nlp-qa-system")
    
    print("Path to dataset files:", path)
    ```
    * by downloading a zip file and transferring the files to the cluster from your local machine with `rsync` or `sftp`.
      
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
   
6. Activate the environment: `conda activate research-project`.


7. Install other dependencies, if any.


8. Running models from the Hugging Face hub requires an access token, 
which you can obtain via the website on your https://huggingface.com profile. 
Save it as an environment variable in bash:
    ```
    export HUGGINGFACE="<<your-token>>"
    ``` 


#### Alternative environment setup with pip3

To get started, log into the Heidelberg University Computational Linguistics cluster and in your home directory:

1. Create a virtual environment: `python3 -m venv venv`
2. Install all dependencies: `pip install -r requirements.txt`
3. Activate environment: `source ~/venv/bin/activate`


### Baseline

#### Running the baseline on the cluster

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


3. Run the script with a config file, for example `baseline_config`:
    ```commandline
    python3 baseline_script.py --config baseline_config
    ```
   If no config specified, the script will run with the default `baseline_config` 
(will through an error of not finding `/workspace/students/reasoning`).
    

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
