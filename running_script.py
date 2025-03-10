from __future__ import annotations

import gc
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from data.DataLoader import DataLoader
from data.DataSaver import DataSaver
from evaluation.Evaluator import MetricEvaluator
from plots.Plotter import Plotter
from prompts.Prompt import Prompt
from settings.Model import Model
from settings.baseline.Baseline import Baseline
from settings.skyline.Skyline import Skyline
from settings.utils import set_device


@hydra.main(version_base=None)
def run_setting(cfg: DictConfig) -> None:
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
    torch.cuda.empty_cache()
    gc.collect()

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

    log_file = sys.stdout

    loader = DataLoader(samples_per_task=cfg.data.samples_per_task)
    data_splits = [split for split, to_use in cfg.data.splits.items() if to_use]
    data_in_splits = {}

    for split in data_splits:
        data_tasks = loader.load_task_data(
            path=cfg.data.path,
            split=split,
            tasks=cfg.data.task_ids,
        )
        data_in_splits[split] = data_tasks

    print("The data is loaded successfully", end="\n\n")

    run_evaluators = defaultdict(dict)
    run_em_accuracies = defaultdict(dict)
    run_sm_accuracies = defaultdict(dict)

    setting = None

    print("The model is being loaded...", end="\n\n")
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
                total_tasks=loader.number_of_tasks,
                total_parts=loader.number_of_parts,
                samples_per_task=loader.samples_per_task,
                prompt=None,
            )
        elif cfg.setting.name == "Skyline":
            model = Model(
                cfg.model.name,
                cfg.model.max_new_tokens,
                cfg.model.temperature,
                cfg.model.to_continue,
            )
            setting = Skyline(
                model=model,
                to_enumerate=cfg.data.to_enumerate,
                total_tasks=loader.number_of_tasks,
                total_parts=loader.number_of_parts,
                samples_per_task=loader.samples_per_task,
                prompt=None,
            )
        elif cfg.setting.name == "Feedback":
            # TODO: add feedback
            pass
        elif cfg.setting.name in ["SD", "SpeculativeDecoding"]:
            # TODO: add speculative decoding
            pass
    except KeyError:
        raise ValueError(
            f"Setting {cfg.setting.name} is not supported. "
            f"Please choose one of the following: Baseline, Skyline, Feedback, SD or SpeculativeDecoding"
        )

    print(f"The model {cfg.model.name} is loaded successfully", flush=True)

    saver = DataSaver(save_to=HydraConfig.get().run.dir)
    print(f"Results will be saved to: {saver.results_path}")

    plotter = Plotter(result_path=saver.results_path)

    for prompt_num, prompt_path in enumerate(cfg.prompt.paths, 1):
        prompt_name = f"prompt_{Path(prompt_path).stem}"
        prompt_evaluator = MetricEvaluator(level="prompt")

        log_file_path, results_file_paths, metrics_file_paths = (
            saver.create_result_paths(prompt_name=prompt_name, splits=data_splits)
        )
        # update result paths
        plotter.result_path = saver.results_path

        # Once the printing is redirected to the log file,
        # the system output will be saved there without additional actions
        if cfg.logging.print_to_file:
            # Print the prompt data to the output file
            # so that it was clear which prompt was used last
            print(f"Starting to query with the prompt: {prompt_name}")
            print(f"Prompt path: {prompt_path}", end="\n\n")
            print(f"Redirecting the system output to: {log_file_path}", flush=True)
            log_file = saver.redirect_printing_to_log_file(log_file_path)

            # Print the config data to the log file
            print(f"Using the model: {cfg.model.name}")

        if cfg.prompt.wrapper:
            prompt = Prompt(
                prompt_path=prompt_path,
                wrapper=cfg.prompt.wrapper,
                name=prompt_name,
            )
        else:
            prompt = Prompt(prompt_path=prompt_path, name=prompt_name)

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

            task_evaluator = MetricEvaluator(level="task")
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
                    task_evaluator=task_evaluator,
                )
                saver.save_task_result(
                    task_id=task_id,
                    task_result=task_result,
                    task_evaluator=task_evaluator,
                    headers=cfg.results.headers,
                    results_path=results_file_paths[split],
                    metrics_path=metrics_file_paths[split],
                )
                print("______________________________", end="\n\n")

            if len(task_evaluator.exact_match_accuracy) != len(tasks):
                raise ValueError(
                    f"Number of tasks and number of accuracies do not match: "
                    f"{len(tasks)} != {len(task_evaluator.exact_match_accuracy)}"
                )

            print(
                f"==> The run for {split.upper()} data is finished successfully <==",
                end="\n\n",
            )
            task_evaluator.print_accuracies(id_=split)

            saver.save_task_accuracy(
                evaluator=task_evaluator,
                accuracy_path=metrics_file_paths[split],
            )

            saver.save_task_metrics(
                evaluator=task_evaluator,
                results_paths=[metrics_file_paths[split], results_file_paths[split]],
            )

            # Plot the prompt accuracies for the split
            plotter.plot_acc_per_task_and_prompt(
                acc_per_prompt_task={
                    "exact_match_accuracy": task_evaluator.exact_match_accuracy,
                    "soft_match_accuracy": task_evaluator.soft_match_accuracy,
                },
                y_label="Accuracies",
                plot_name_add=f"{prompt_name}_{split}_",
            )

            prompt_evaluator.update(task_evaluator)

            run_evaluators[split][prompt] = task_evaluator
            run_em_accuracies[split][prompt_name] = task_evaluator.exact_match_accuracy
            run_sm_accuracies[split][prompt_name] = task_evaluator.soft_match_accuracy

        print("\n- RUN RESULTS -", end="\n\n")

        prompt_evaluator.print_metrics(id_=prompt_name)

        print(
            "Processed",
            loader.number_of_tasks,
            "tasks in total with",
            loader.samples_per_task,
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

    if len(cfg.prompt.paths) > 1:
        for split in data_in_splits.keys():
            saver.save_split_accuracy(
                task_ids=loader.tasks,
                prompt_evaluators=run_evaluators[split],
                split=split,
            )
            # Plot accuracies for all prompts the model ran with
            plotter.result_path = saver.run_path
            plotter.plot_accuracies(
                accuracies=run_em_accuracies[split],
                soft_match_accuracies=run_sm_accuracies[split],
                additional_info=f"{split}_",
                compare_prompts=True,
            )

    print("Plots are saved successfully and general accuracies are saved", end="\n\n")

    print("The script has finished running successfully")


if __name__ == "__main__":
    run_setting()
