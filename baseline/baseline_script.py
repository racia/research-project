from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(Path.cwd()).parents[0]))
from data.DataSaver import DataSaver
from data.DataProcessor import DataProcessor
from data.DataLoader import DataLoader
from data.Statistics import Statistics as St
from Model import Baseline
from baseline.config.baseline_config import Config


def run_model(cfg: Config | DictConfig) -> None:
    """
    The function to run a model without modifications
    over the task data with the given config with such steps:
    1. initiate instances of model and data classes
    2. [optional] redirect system output into the logging file
    3. load the model and the data, set the configurations
    4. loop through all desired tasks of desired data splits
    5. save the results after each cycle
    6. report and save accuracy for the run
    7. [optional*] return the system output in place

    :param cfg: config instance
    """
    print("Config data for the run:", cfg)
    print("Running the script...")

    saver = DataSaver()
    processor = DataProcessor()
    loader = DataLoader()
    results_path = cfg.repository.path + cfg.results.path + cfg.prompt.name
    log_file = sys.stdout

    if cfg.results.print_to_file:
        log_file, results_path = saver.redirect_printing_to_file(path=results_path)

    saver.set_results_details(results_path=results_path, headers=cfg.results.headers)

    model = Baseline(
        model_name=cfg.model.name,
        max_new_tokens=cfg.model.max_new_tokens,
        temperature=cfg.model.temperature,
        log_file=log_file,
    )

    model.load_model()
    model.set_system_prompt(prompt=cfg.prompt.text)
    print(f"The model {cfg.model.name} is loaded successfully", file=log_file)

    model.total_tasks = 0
    data_in_splits = {}

    for split, to_fetch in cfg.data.splits.items():
        if to_fetch:
            data_tasks = loader.load_data(
                path=cfg.data.path, split=split, tasks=cfg.data.task_ids
            )
            processed_data = processor.process_data(data=data_tasks)
            model.total_tasks += len(data_tasks)
            data_in_splits[split] = processed_data

    print("The data is loaded successfully", end="\n\n", file=log_file)
    print("Starting to query the model", end="\n\n", file=log_file)

    for split, tasks in data_in_splits.items():
        for task_id, task in tasks.items():
            task_result = model.iterate_task(
                task_id=task_id,
                task_data=task,
                no_samples=cfg.data.samples_per_task,
                to_enumerate=cfg.data.to_enumerate,
                to_continue=cfg.model.to_continue,
                parse_output=cfg.results.parse,
            )
            saver.save_output(data=task_result)
            print("______________________________", end="\n\n", file=log_file)

    print("The run is finished successfully", file=log_file)

    print("\n- RUN RESULTS -", end="\n\n", file=log_file)

    print(
        "Processed",
        model.total_tasks,
        "tasks in total with",
        cfg.data.samples_per_task,
        "samples in each",
        file=log_file,
    )
    print(
        "Total samples processed",
        model.total_tasks * cfg.data.samples_per_task,
        end="\n\n",
        file=log_file,
    )

    model.accuracy = round(accuracy_score(model.y_true, model.y_pred), 2)
    print("General accuracy:", model.accuracy, file=log_file)

    model.soft_match_accuracy = round(
        St.soft_match_accuracy_score(model.y_true, model.y_pred), 2
    )
    print("General soft match accuracy:", model.soft_match_accuracy, file=log_file)

    row = [
        {"accuracy": model.accuracy, "soft_match_accuracy": model.soft_match_accuracy}
    ]
    saver.save_output(data=row)

    if cfg.results.print_to_file:
        # console printing must be returned
        # if printing was redirected to logs created at the beginning of the script
        # 'log_file' will exist in that case as well
        saver.return_console_printing(log_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        help="use the settings from the config file of given name "
        "(with relative path from the config directory)",
        metavar="CONFIG",
    )
    args = parser.parse_args()

    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)

    with initialize(version_base=None, config_path="config/"):
        if args.config:
            config = compose(config_name=args.config)
        else:
            # for cases of running the script in the IDE
            config = compose(config_name="baseline_config")

    run_model(cfg=config)
