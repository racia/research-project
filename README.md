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

## Structure

* [data](./data)
    * golden evaluation dataset
    * methods for data processing
    * methods to load and save data

* [prompts](./inference) — the prompts used in the experiments as text files

* [settings](./settings)
    * [baseline](./settings/baseline) — contains the baseline setting
    * [skyline](./settings/skyline) — contains the skyline setting
    * [feedback](./settings/feedback) — contains the feedback setting
    * [speculative-decoding](./settings/speculative-decoding) — contains the SD setting

* [interpretability](./interpretability) — methods

* [evaluation](./evaluation) - contains everything related to evaluation

## Getting started

This project can be run by using the provided bash file `setting_bwUniCluster.sh` or `setting_CLuster.sh`. For more
detailed information about the general setup, please refer to [SETUP.md](SETUP.md).

## Settings

This project focuses on two approaches aimed at improving chain-of-thought reasoning: "Feedback" and "Speculative
Decoding."
For more detailed information about the settings, please refer to this [README.md](settings/README.md).

To run one of the settings, you can use the following command:

```bash
bash <setting_name>.sh
```

## Results

At some point, we will hopefully have some results that we can summarise here.