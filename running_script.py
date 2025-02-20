from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from data.DataLoader import DataLoader
from data.DataSaver import DataSaver
from data.Statistics import Statistics
from plots.Plotter import Plotter
from prompts.Prompt import Prompt
from settings.Model import Model
from settings.baseline.Baseline import Baseline
from settings.utils import set_device


@hydra.main(version_base=None)
def run_model(cfg: DictConfig) -> None:
    """
    The function to run a model without modifications
    over the task data with the given config with such steps:
    1. initiate instances of model and data classes
    2. [optional] redirect system output into the logging file
    3. load the model and the data, set the configurations
    4. loop through all desired tasks of desired data splits
    5. save the results after each cycle
    6. report and save accuracy for the run
    7. [optional] return the system output in place

    :param cfg: config instance
    :return: None
    """
    OmegaConf.resolve(cfg)

    set_device()

    print(
        "Config data for the run:",
        OmegaConf.to_yaml(cfg),
        end="\n\n",
        flush=True,
        sep="\n",
    )
    print("Running the script...")

    loader = DataLoader(samples_per_task=cfg.data.samples_per_task)
    log_file = sys.stdout

    saver = DataSaver(save_to=HydraConfig.get().run.dir)
    plotter = Plotter(result_path=saver.results_path)
    os.environ["INTERP_DIR"] = os.path.join("/research-project/", cfg.interpretability.path)

    stats = Statistics()
    strict_accuracies = {}
    soft_match_accuracies = {}

    try:
        if cfg.setting.name == "Baseline":
            model = Model(
                cfg.model.name,
                cfg.model.max_new_tokens,
                cfg.model.temperature,
                cfg.model.to_continue,
            )
            setting = Baseline(
                model=model,
                to_enumerate=cfg.data.to_enumerate,
                parse_output=cfg.results.parse,
                statistics=stats,
                prompt=None,
                samples_per_task=cfg.data.samples_per_task,
            )
        elif cfg.setting.name == "Skyline":
            model = Model(
                cfg.model_name, cfg.max_new_tokens, cfg.temperature, cfg.to_continue
            )

            setting = Baseline(
                model=model,
                to_enumerate=cfg.data.to_enumerate,
                parse_output=cfg.results.parse,
                statistics=stats,
                prompt=None,
                samples_per_task=cfg.data.samples_per_task,
            )
        elif cfg.setting.name == "Feedback":
            # TODO: add feedback
            pass
        elif cfg.setting.name == "SD" or "SpeculativeDecoding":
            # TODO: add speculative decoding
            pass
    except KeyError:
        raise ValueError(
            f"Setting {cfg.setting} is not supported. "
            f"Please choose one of the following: Baseline, Skyline, Feedback, SD"
        )

    print("The model is being loaded...", end="\n\n")
    setting.model.total_tasks = 0
    data_in_splits = {}
    print(f"The model {cfg.model.name} is loaded successfully", flush=True)

    data_splits = [split for split, to_use in cfg.data.splits.items() if to_use]

    for split in data_splits:
        data_tasks = loader.load_task_data(
            path=cfg.data.path,
            split=split,
            tasks=cfg.data.task_ids,
        )
        data_in_splits[split] = data_tasks
        setting.model.total_tasks += len(data_tasks)
    setting.model.total_parts = loader.number_of_parts
    print("The data is loaded successfully", end="\n\n")

    for prompt_num, prompt_path in enumerate(cfg.prompt.paths, 1):
        prompt_name = f"prompt_{Path(prompt_path).stem}"
        log_file_path, results_file_paths, accuracy_file_paths = (
            saver.create_result_paths(prompt_name=prompt_name, splits=data_splits)
        )
        plotter.result_path = saver.results_path / prompt_name

        # Once the printing is redirected to the log file,
        # the system output will be saved there without additional actions
        if cfg.logging.print_to_file:
            # Print the prompt data to the output file
            # so that it was clear which prompt was used last
            print(f"Starting to query with the prompt: {prompt_name}")
            print(f"Prompt path: {prompt_path}", end="\n\n")
            print(
                f"Redirecting the system output to: {log_file_path}",
                flush=True,
            )

            log_file = saver.redirect_printing_to_log_file(log_file_path)
            setting.model.log_file = log_file

            # Print the config data to the log file
            print(f"Using the model: {cfg.model.name}")

        if cfg.prompt.wrapper:
            prompt = Prompt(
                prompt_path=prompt_path,
                wrapper=cfg.prompt.wrapper,
            )
        else:
            prompt = Prompt(prompt_path=prompt_path)

        setting.prompt = prompt

        print(f"Prompt: {prompt_name}, path: {prompt_path}")

        print("- THE SYSTEM SPROMPT -")
        print("______________________________")
        print(prompt.text)
        print("______________________________", end="\n\n")
        setting.question_id = 0

        for split, tasks in data_in_splits.items():
            print(
                f"Starting to query the model with {split.upper()} data...", end="\n\n"
            )
            if split not in strict_accuracies.keys():
                strict_accuracies[split] = {}
                soft_match_accuracies[split] = {}

            setting.accuracies_per_task = []
            setting.soft_match_accuracies_per_task = []

            for task_id, task in sorted(tasks.items()):
                if cfg.prompt.examples.to_add:
                    setting.prompt.add_examples(
                        task_id=task_id, example_config=cfg.prompt.examples
                    )
                else:
                    setting.prompt.use_original_prompt()

                task_result = setting.iterate_task(
                    task_id=task_id,
                    task_data=task,
                    prompt_name=f"'{prompt_name}' {prompt_num}/{len(cfg.prompt.paths)}",
                    interpr = cfg.interpretability
                )
                saver.save_output(
                    data=task_result,
                    headers=cfg.results.headers,
                    file_path=results_file_paths[split],
                )
                print("______________________________", end="\n\n")

            if len(setting.accuracies_per_task) != len(tasks):
                raise ValueError(
                    f"Number of tasks and number of accuracies do not match: "
                    f"{len(tasks)} != {len(setting.accuracies_per_task)}"
                )

            print(
                f"==> The run for {split.upper()} data is finished successfully <==",
                end="\n\n",
            )

            # Prepare the prompt accuracies for saving
            accuracies_to_save = []
            for task_id, accuracy, soft_match_accuracy in zip(
                tasks.keys(),
                setting.accuracies_per_task,
                setting.soft_match_accuracies_per_task,
            ):
                accuracies_to_save.append(
                    {
                        "task": task_id,
                        "accuracy": accuracy,
                        "soft_match_accuracy": soft_match_accuracy,
                    }
                )
            # "zero task" stores the mean accuracy for all tasks
            accuracies_to_save.append(
                {
                    "task": 0,
                    "accuracy": statistics.mean(setting.accuracies_per_task),
                    "soft_match_accuracy": statistics.mean(
                        setting.soft_match_accuracies_per_task
                    ),
                }
            )

            # Save the prompt accuracies for the split
            saver.save_output(
                data=accuracies_to_save,
                headers=["task", "accuracy", "soft_match_accuracy"],
                file_path=accuracy_file_paths[split],
            )

            # Save prompt accuracies generally
            strict_accuracies[split][prompt_name] = setting.accuracies_per_task
            soft_match_accuracies[split][
                prompt_name
            ] = setting.soft_match_accuracies_per_task

            plotter.plot_acc_per_task_and_prompt(
                acc_per_prompt_task={
                    "accuracy": setting.accuracies_per_task,
                    "soft_match_accuracy": setting.soft_match_accuracies_per_task,
                },
                y_label="Accuracies",
                plot_name_add=f"{prompt_name}_{split}_",
            )

        print("\n- RUN RESULTS -", end="\n\n")

        print(
            "Processed",
            setting.total_tasks,
            "tasks in total with",
            cfg.data.samples_per_task,
            "samples in each",
        )
        print(
            "Total samples processed",
            setting.total_tasks * cfg.data.samples_per_task,
            end="\n\n",
        )

        if cfg.logging.print_to_file:
            # console printing must be returned
            # if printing was redirected to logs created at the beginning of the script
            # 'log_file' will exist in that case as well
            saver.return_console_printing(log_file)

        print("The run is finished successfully")
        print("______________________________")

    #plotter.result_path = saver.results_path

    if len(cfg.prompt.paths) > 1:
        for split in data_in_splits.keys():
            # Plot accuracies for all prompts the model ran with
            plotter.plot_accuracies(
                accuracies=strict_accuracies[split],
                soft_match_accuracies=soft_match_accuracies[split],
                additional_info=f"{split}_",
                compare_prompts=True,
            )

            gen_accuracies_to_save = {}
            gen_headers = ["task_id"]

            for prompt_name, strict_accuracies, soft_match_accuracies in zip(
                strict_accuracies[split].keys(),
                strict_accuracies[split].values(),
                soft_match_accuracies[split].values(),
            ):
                prompt_name_ = prompt_name.replace("prompt_", "")
                prompt_headers = {
                    "strict": f"{prompt_name_}_strict_accuracy",
                    "soft_match": f"{prompt_name_}_soft_match_accuracy",
                }
                for task_id, (
                    strict_accuracy,
                    soft_match_accuracy,
                ) in enumerate(zip(strict_accuracies, soft_match_accuracies), 1):

                    if gen_accuracies_to_save.get(task_id) is None:
                        gen_accuracies_to_save[task_id] = {"task_id": task_id}

                    gen_accuracies_to_save[task_id].update(
                        {
                            prompt_headers["strict"]: strict_accuracy,
                            prompt_headers["soft_match"]: soft_match_accuracy,
                        }
                    )
                if gen_accuracies_to_save.get(0) is None:
                    gen_accuracies_to_save[0] = {"task_id": 0}

                gen_accuracies_to_save[0].update(
                    {
                        prompt_headers["strict"]: statistics.mean(strict_accuracies),
                        prompt_headers["soft_match"]: statistics.mean(
                            soft_match_accuracies
                        ),
                    }
                )
                gen_headers.extend(prompt_headers.values())

            for task_id, accuracies in gen_accuracies_to_save.items():
                gen_accuracies_to_save[task_id].update(
                    {
                        "mean_strict_accuracy": statistics.mean(
                            [
                                value
                                for name, value in accuracies.items()
                                if "strict" in name
                            ]
                        ),
                        "mean_soft_match_accuracy": statistics.mean(
                            [
                                value
                                for name, value in accuracies.items()
                                if "soft_match" in name
                            ]
                        ),
                    }
                )

            gen_headers.extend(["mean_strict_accuracy", "mean_soft_match_accuracy"])

            saver.save_output(
                data=list(gen_accuracies_to_save.values()),
                headers=gen_headers,
                file_path=saver.run_results_path / f"{split}_accuracies.csv",
            )

    print("Plots are saved successfully and general accuracies are saved", end="\n\n")

    print("The script has finished running successfully")


if __name__ == "__main__":
    run_model()
