from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

sys.path.insert(0, str(Path(Path.cwd()).parents[0]))
from data.DataSaver import DataSaver
from data.DataLoader import DataLoader
from data.Statistics import Statistics
from data.Plotter import Plotter
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
    :return: None
    """
    print("Config data for the run:", cfg)
    print("Running the script...")

    loader = DataLoader()
    log_file = sys.stdout
    saver = DataSaver()
    stats = Statistics()
    plotter = Plotter(result_path=str(saver.results_path))

    all_accuracies = {
        "strict": {},
        "soft_match": {},
    }

    model = Baseline(
        model_name=cfg.model.name,
        max_new_tokens=cfg.model.max_new_tokens,
        temperature=cfg.model.temperature,
        log_file=log_file,
        statistics=stats,
    )

    model.load_model()
    model.total_tasks = 0
    data_in_splits = {}

    for split, to_fetch in cfg.data.splits.items():
        if to_fetch:
            data_tasks = loader.load_data(
                path=cfg.data.path, split=split, tasks=cfg.data.task_ids
            )
            model.total_tasks += len(data_tasks)
            data_in_splits[split] = data_tasks

    for prompt_name, prompt_path in zip(cfg.prompt.names, cfg.prompt.paths):
        log_file_path, results_file_path, accuracy_file_paths = saver.create_path_files(
            results_path=Path(cfg.repository.path, cfg.results.path),
            prompt_name=prompt_name,
        )

        if cfg.results.print_to_file:
            log_file = saver.redirect_printing_to_log_file(log_file_path)
            model.log_file = log_file

        print(f"The model {cfg.model.name} is loaded successfully", file=log_file)
        print("The data is loaded successfully", end="\n\n", file=log_file)
        print(f"Prompt: {prompt_name}, path: {prompt_path}", file=log_file)
        print("Starting to query the model", end="\n\n", file=log_file)

        model.set_system_prompt(prompt_file_path=prompt_path)

        for split, tasks in data_in_splits.items():
            model.accuracies_per_task = []
            model.soft_match_accuracies_per_task = []
            if split not in all_accuracies.keys():
                all_accuracies[split] = {}
            all_accuracies[split][prompt_name] = {}

            for task_id, task in sorted(tasks.items()):
                task_result = model.iterate_task(
                    task_id=task_id,
                    task_data=task,
                    no_samples=cfg.data.samples_per_task,
                    to_enumerate=cfg.data.to_enumerate,
                    to_continue=cfg.model.to_continue,
                    parse_output=cfg.results.parse,
                )
                saver.save_output(
                    data=task_result,
                    headers=cfg.results.headers,
                    file_path=results_file_path,
                )
                print("______________________________", end="\n\n", file=log_file)

            print(f"The run for {split} data is finished successfully", file=log_file)

            all_accuracies = {
                "prompt": {
                    "split": {"task": {"accuracy": 0, "soft_match_accuracy": 0}}
                },
            }

            accuracies_to_save = []
            for task_id, accuracy, soft_match_accuracy in zip(
                tasks.keys(),
                model.accuracies_per_task,
                model.soft_match_accuracies_per_task,
            ):
                accuracies_to_save.append(
                    {
                        "task": task_id,
                        "accuracy": accuracy,
                        "soft_match_accuracy": soft_match_accuracy,
                    }
                )

            saver.save_output(
                data=accuracies_to_save,
                headers=["task", "accuracy", "soft_match_accuracy"],
                file_path=accuracy_file_paths[str(split)],
            )

            plotter.plot_acc_per_task(
                acc_per_task=model.accuracies_per_task,
                y_label="Accuracy",
                plot_name_add=f"{prompt_name}_{str(split)}_",
            )
            plotter.plot_acc_per_task(
                acc_per_task=model.soft_match_accuracies_per_task,
                y_label="Soft Match Accuracy",
                plot_name_add=f"{prompt_name}_{str(split)}_",
            )

            all_accuracies["strict"][str(split)][prompt_name] = [
                task["accuracy"] for task in accuracies_to_save
            ]
            all_accuracies["soft_match"][str(split)][prompt_name] = [
                task["soft_match_accuracy"] for task in accuracies_to_save
            ]

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

        model.accuracy = round(stats.accuracy_score(model.y_true, model.y_pred), 2)
        print("General accuracy:", model.accuracy, file=log_file)

        model.soft_match_accuracy = round(
            stats.soft_match_accuracy_score(model.y_true, model.y_pred), 2
        )
        print("General soft match accuracy:", model.soft_match_accuracy, file=log_file)

        row = [
            {
                "accuracy": model.accuracy,
                "soft_match_accuracy": model.soft_match_accuracy,
            }
        ]
        saver.save_output(
            data=row,
            headers=["accuracy", "soft_match_accuracy"],
            file_path=results_file_path,
        )

        if cfg.results.print_to_file:
            # console printing must be returned
            # if printing was redirected to logs created at the beginning of the script
            # 'log_file' will exist in that case as well
            saver.return_console_printing(log_file)

    # Plot accuracies for all prompts the model ran with
    if len(cfg.prompt.names) > 1:
        for split in data_in_splits.keys():
            plotter.plot_acc_per_task_and_prompt(
                acc_per_prompt_task=all_accuracies["strict"][split],
                y_label="Accuracy",
                plot_name_add=f"{str(split)}_",
            )
            plotter.plot_acc_per_task_and_prompt(
                acc_per_prompt_task=all_accuracies["soft_match"][split],
                y_label="Soft Match Accuracy",
                plot_name_add=f"{str(split)}_",
            )


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
