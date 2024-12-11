from __future__ import annotations

import os
import sys
from pathlib import Path
from argparse import ArgumentParser

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(Path.cwd()).parents[0]))
from data.DataHandler import DataHandler
from data.Statistics import Statistics as St
import Model
from baseline.config.baseline_config import BaselineConfig


def run(cfg: BaselineConfig, model) -> None:
    print("Config data for the run:", cfg)

    data = DataHandler()
    results_path = cfg.repository.path + cfg.results.path + cfg.prompt.name

    if cfg.results.print_to_file:
        log_file, results_path = data.redirect_printing_to_file(path=results_path)

    data.set_results_details(results_path=results_path, headers=cfg.results.headers)

    # model.load_model()
    model.set_system_prompt(prompt=cfg.prompt.text)
    print("The model is loaded successfully")

    model.total_tasks = 0
    data_in_splits = {}

    for split, to_fetch in cfg.data.splits.items():
        if to_fetch:
            data_tasks = data.read_data(path=cfg.data.path, split=split,
                                        tasks=cfg.data.task_ids)
            processed_data = data.process_data(data=data_tasks)
            model.total_tasks += len(data_tasks)
            data_in_splits[split] = processed_data

    print("The data is loaded successfully", end="\n\n")
    print("Starting to query the model", end="\n\n")

    model.to_enumerate = cfg.data.to_enumerate
    for split, tasks in data_in_splits.items():
        for task_id, task in tasks.items():
            task_result = model.iterate_task(
                task_id=task_id, task_data=task,
                no_samples=cfg.data.samples_per_task
            )
            data.save_output(data=task_result)
            print("______________________________", end="\n\n")

    print("The run is finished successfully")

    print("\n- RUN RESULTS -", end="\n\n")

    print("Processed", model.total_tasks, "tasks in total with",
          cfg.data.samples_per_task, "samples in each")
    print("Total samples processed",
          model.total_tasks * cfg.data.samples_per_task, end="\n\n")

    model.accuracy = round(accuracy_score(model.y_true, model.y_pred), 2)
    print("General accuracy:", model.accuracy)

    model.soft_match_accuracy = round(St.soft_match_accuracy_score(model.y_true, model.y_pred), 2)
    print("General soft accuracy:", model.soft_match_accuracy)

    row = [{"accuracy": model.accuracy,
            "soft_match_accuracy": model.soft_match_accuracy}]
    data.save_output(data=row)

    if cfg.results.print_to_file:
        # console printing must be returned
        # if printing was redirected to logs created at the beginning of the script
        # 'log_file' will exist in that case as well
        data.return_console_printing(log_file)


if __name__ == "__main__":
    # TODO: add to config if we want to print certain data
    #  (level of messages: INFO, CRITICAL and so on)

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", dest="config",
                        help="use the settings from the config file of given name "
                             "(with relative path from the config directory)",
                        metavar="CONFIG")
    args = parser.parse_args()

    cs = ConfigStore.instance()
    cs.store(name="config", node=BaselineConfig)

    with initialize(version_base=None, config_path="config"):
        if args.config:
            cfg = compose(config_name=args.config)
        else:
            # for cases of running the script interactively
            cfg = compose(config_name="baseline_config")

        baseline = Model.Baseline(
            model_name=cfg.model.name,
            max_new_tokens=cfg.model.max_new_tokens,
            temperature=cfg.model.temperature)
        print(os.getcwd())
        run(cfg=cfg, model=baseline)