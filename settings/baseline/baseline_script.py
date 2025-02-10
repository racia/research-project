from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from settings.baseline.utils import set_device

sys.path.insert(0, str(Path(Path.cwd()).parents[0]))

from prompts.Prompt import Prompt
from data.DataSaver import DataSaver
from data.DataLoader import DataLoader
from data.Statistics import Statistics
from plots.Plotter import Plotter
from Baseline import Baseline


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
    7. [optional*] return the system output in place

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

    if cfg.repository.save:
        saver = DataSaver(
            project_dir=cfg.repository.path,
            subproject_dir=cfg.results.path,
            save_to_repo=True,
        )
    else:
        saver = DataSaver(
            project_dir=HydraConfig.get().runtime.output_dir,
            save_to_repo=False,
        )
    print(f"Results will be saved to: {saver.run_results_path}")
    plotter = Plotter(result_path=saver.run_results_path)
    os.environ["OUTPUT_DIR"] = str(saver.run_results_path)
    os.environ["INTERP_DIR"] = cfg.repository.path+cfg.interpretability.path

    stats = Statistics()
    strict_accuracies = {}
    soft_match_accuracies = {}

    model = Baseline(
        model_name=cfg.model.name,
        max_new_tokens=cfg.model.max_new_tokens,
        temperature=cfg.model.temperature,
        statistics=stats,
        samples_per_task=cfg.data.samples_per_task,
        to_enumerate=cfg.data.to_enumerate,
        to_continue=cfg.model.to_continue,
        parse_output=cfg.results.parse,
    )

    print("The model is being loaded...", end="\n\n")
    model.load_model()
    model.total_tasks = 0
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
        model.total_tasks += len(data_tasks)
    model.total_parts = loader.number_of_parts
    print("The data is loaded successfully", end="\n\n")

    for prompt_num, prompt_path in enumerate(cfg.prompt.paths, 1):
        prompt_name = f"prompt_{Path(prompt_path).stem}"
        log_file_path, results_file_paths, accuracy_file_paths = (
            saver.create_result_paths(prompt_name=prompt_name, splits=data_splits)
        )
        plotter.result_path = saver.run_results_path / prompt_name

        # Once the printing is redirected to the log file,
        # the system output will be saved there without additional actions
        if cfg.results.print_to_file:
            # Print the prompt data to the output file
            # so that it was clear which prompt was used last
            print(f"Starting to query with the prompt: {prompt_name}")
            print(f"Prompt path: {prompt_path}", end="\n\n")
            print(
                f"Redirecting the system output to the log file: {log_file_path}",
                flush=True,
            )

            log_file = saver.redirect_printing_to_log_file(log_file_path)
            model.log_file = log_file

            # Print the config data to the log file
            print(f"Using the model: {cfg.model.name}")

        if cfg.prompt.wrapper:
            prompt = Prompt(
                prompt_path=prompt_path,
                wrapper=cfg.prompt.wrapper,
            )
        else:
            prompt = Prompt(prompt_path=prompt_path)
        model.prompt = prompt
        print(f"Prompt: {prompt_name}, path: {prompt_path}")

        print("- THE SYSTEM SPROMPT -")
        print("______________________________")
        print(prompt.text)
        print("______________________________", end="\n\n")
        model.question_id = 0

        for split, tasks in data_in_splits.items():
            print(
                f"Starting to query the model with {split.upper()} data...", end="\n\n"
            )
            if split not in strict_accuracies.keys():
                strict_accuracies[split] = {}
                soft_match_accuracies[split] = {}

            model.accuracies_per_task = []
            model.soft_match_accuracies_per_task = []

            for task_id, task in sorted(tasks.items()):
                task_result = model.iterate_task(
                    task_id=task_id,
                    task_data=task,
                    prompt_name=f"'{prompt_name}' {prompt_num}/{len(cfg.prompt.names)}",
                    interpr = cfg.interpretability
                )
                saver.save_output(
                    data=task_result,
                    headers=cfg.results.headers,
                    file_path=results_file_paths[split],
                )
                print("______________________________", end="\n\n")

            if len(model.accuracies_per_task) != len(tasks):
                raise ValueError(
                    f"Number of tasks and number of accuracies do not match: "
                    f"{len(tasks)} != {len(model.accuracies_per_task)}"
                )

            print(
                f"==> The run for {split.upper()} data is finished successfully <==",
                end="\n\n",
            )

            # Prepare the prompt accuracies for saving
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
            # "zero task" stores the mean accuracy for all tasks
            accuracies_to_save.append(
                {
                    "task": 0,
                    "accuracy": statistics.mean(model.accuracies_per_task),
                    "soft_match_accuracy": statistics.mean(
                        model.soft_match_accuracies_per_task
                    ),
                }
            )

            # Save the prompt accuracies for the split
            saver.save_output(
                data=accuracies_to_save,
                headers=["task", "accuracy", "soft_match_accuracy"],
                file_path=accuracy_file_paths[split],
            )

            # Store prompt accuracies generally to save later together
            strict_accuracies[split][prompt_name] = model.accuracies_per_task
            soft_match_accuracies[split][
                prompt_name
            ] = model.soft_match_accuracies_per_task

            # Plot accuracies for the prompt
            plotter.plot_accuracies(
                accuracies=model.accuracies_per_task,
                soft_match_accuracies=model.soft_match_accuracies_per_task,
                additional_info=f"{prompt_name}_{split}",
                compare_prompts=False,
            )

        print("\n- RUN RESULTS -", end="\n\n")

        print(
            "Processed",
            model.total_tasks,
            "tasks in total with",
            cfg.data.samples_per_task,
            "samples in each",
        )
        print(
            "Total samples processed",
            model.total_tasks * cfg.data.samples_per_task,
            end="\n\n",
        )

        if cfg.results.print_to_file:
            # console printing must be returned
            # if printing was redirected to logs created at the beginning of the script
            # 'log_file' will exist in that case as well
            saver.return_console_printing(log_file)

        print("The run is finished successfully")
        print("______________________________")

    plotter.result_path = saver.run_results_path

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
