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
To get started, log into the Heidelberg University Computational Linguistics cluster and in your home directory:

1. Create a virtual environment: `python3 -m venv venv`
2. Install all dependencies: `pip install -r requirements.txt`
3. Activate environment: `source ~/venv/bin/activate`

After git cloning the repository, change direcory to it: `cd research-project/`

#### ! Important

Please note, that the scripts expect the bAbI data on your home directory of the uni cluster, e.g. `~/tasks_1-20_v1-2/`

### Baseline

#### Getting started

1. Create a virtual environment: `python3 -m venv venv`
2. Install all dependencies: `pip install -r requirements.txt`
3. Activate environment: `source ~/venv/bin/activate`

After git cloning the repository, change direcory to it. `cd research-project/`

#### ! Important

Please note, that the scripts expect the bAbI data on your home directory of the uni cluster, e.g. `~/tasks_1-20_v1-2/`


### Baseline

#### Running the baseline

Running models from the Hugging Face hub requires an access token, which you can obtain via the website on
your https://huggingface.com profile.

1. Save your token as an environment variable in bash:

```
export HUGGINGFACE="<<your-token>>"
```

2. Change directory to the baseline folder: `cd /baseline`
3. Submit the batch job: `sbatch initial_baseline.sh`, which will run script.py and save the outputs to "init_bl.txt"

### Data

### Prompts

### Settings

#### Feedback

#### Speculative Decoding

### Interpretability

### Prompting Experiments

### Evaluation

### Fine-Tuning

### Tests
