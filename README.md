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
