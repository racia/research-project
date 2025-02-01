# Setting up the project

As setting up this project is not straightforward, this README includes a detailed guide on how to set up the project on
both "CLuster" and "bwUniCluster".

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