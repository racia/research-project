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
from inference.DataLevels import Split
from inference.Prompt import Prompt
from interpretability.Interpretability import Interpretability
from plots.Plotter import Plotter
from settings.Model import Model
from settings.SD.SpeculativeDecoding import SpeculativeDecoding
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

    saver = DataSaver(save_to=HydraConfig.get().run.dir)
    print(f"Results will be saved to: {saver.results_path}")
    plotter = Plotter(results_path=saver.results_path)

    run_evaluators = defaultdict(dict)
    run_em_accuracies = defaultdict(dict)
    run_sm_accuracies = defaultdict(dict)
    run_em_std = defaultdict(dict)
    run_sm_std = defaultdict(dict)

    setting = None

    if not hasattr(cfg.setting, "name"):
        raise ValueError("The setting name is not provided in the config file")

    print("The model is being loaded...", end="\n\n")
    if hasattr(cfg, "model"):
        model = Model(
            cfg.model.name,
            cfg.model.max_new_tokens,
            cfg.model.temperature,
            cfg.model.to_continue,
            cfg.model.mode,
        )
    elif hasattr(cfg, "student"):
        model = Model(
            cfg.student.name,
            cfg.student.max_new_tokens,
            cfg.student.temperature,
            cfg.student.to_continue,
            cfg.student.mode,
        )
    else:
        raise ValueError("No base model is provided in the config.")

    # Load scenery words
    scenery_words = loader.load_scenery()

    interpretability = (
        Interpretability(
            model=model,
            plotter=plotter,
            save_heatmaps=cfg.results.save_heatmaps,
            scenery_words=scenery_words,
        )
        if cfg.setting.interpretability
        else None
    )

    multi_system = False

    if cfg.setting.name == "Baseline":
        setting = Baseline(
            model=model,
            to_enumerate=cfg.data.to_enumerate,
            total_tasks=loader.number_of_tasks,
            total_parts=loader.number_of_parts,
            samples_per_task=loader.samples_per_task,
            init_prompt=None,
            wrapper=cfg.data.wrapper if cfg.data.wrapper else None,
            interpretability=interpretability,
        )
    elif cfg.setting.name == "Skyline":
        setting = Skyline(
            model=model,
            to_enumerate=cfg.data.to_enumerate,
            total_tasks=loader.number_of_tasks,
            total_parts=loader.number_of_parts,
            samples_per_task=loader.samples_per_task,
            init_prompt=None,
            wrapper=cfg.data.wrapper if cfg.data.wrapper else None,
            interpretability=interpretability,
        )
    elif cfg.setting.name == "Feedback":
        # TODO: add feedback
        pass
    elif cfg.setting.name in ["SD", "SpeculativeDecoding"]:
        multi_system = True
        teacher = Model(
            cfg.teacher.name,
            cfg.teacher.max_new_tokens,
            cfg.teacher.temperature,
            cfg.teacher.to_continue,
            cfg.teacher.mode,
        )
        setting = SpeculativeDecoding(
            student=model,
            teacher=teacher,
            to_enumerate=cfg.data.to_enumerate,
            total_tasks=loader.number_of_tasks,
            total_parts=loader.number_of_parts,
            init_prompt=None,
            eval_prompt=None,
            resume_prompt=None,
            samples_per_task=cfg.data.samples_per_task,
            wrapper=cfg.data.wrapper if cfg.data.wrapper else None,
            interpretability=interpretability,
        )
    else:
        raise ValueError(
            f"Setting {cfg.setting.name} is not supported. "
            f"Please choose one of the following: Baseline, Skyline, Feedback, SD or SpeculativeDecoding"
        )

    setting.total_tasks = 0

    for prompt_num, prompt_path in enumerate(cfg.init_prompt.paths, 1):
        prompt_name = f"prompt_{Path(prompt_path).stem}"
        prompt_evaluator_before = (
            MetricEvaluator(level="prompt") if multi_system else None
        )
        prompt_evaluator_after = MetricEvaluator(level="prompt")

        log_file_name, results_file_names, metrics_file_names = (
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
            print(f"Redirecting the system output to: {log_file_name}", flush=True)
            log_file = saver.redirect_printing_to_log_file(log_file_name)

            # Print the config data to the log file
            if cfg.setting.name in ["Baseline", "Skyline"]:
                print(f"Using the model: {cfg.model.name}")
            elif cfg.setting.name in ["SD", "SpeculativeDecoding", "Feedback"]:
                print(f"Using the student model: {cfg.student.name}")
                print(f"Using the teacher model: {cfg.teacher.name}")

        init_prompt = Prompt(prompt_path=prompt_path, name=prompt_name)
        setting.init_prompt = init_prompt

        print(f"Prompt: {prompt_name}, path: {prompt_path}")

        print("- THE SYSTEM SPROMPT -")
        print("______________________________")
        print(init_prompt.text)
        print("______________________________", end="\n\n")

        if cfg.setting.name in ["SD", "SpeculativeDecoding"]:
            setting.eval_prompt = Prompt(
                prompt_path=cfg.eval_prompt.paths[0], wrapper=cfg.eval_prompt.wrapper
            )

            print("- THE EVAL PROMPT -")
            print("______________________________")
            print(setting.eval_prompt.text)
            print("______________________________", end="\n\n", flush=True)

            setting.resume_prompt = Prompt(
                prompt_path=cfg.resume_prompt.paths[0],
                wrapper=cfg.resume_prompt.wrapper,
            )

            print("- THE RESUME PROMPT -")
            print("______________________________")
            print(setting.resume_prompt.text)
            print("______________________________", end="\n\n", flush=True)

        setting.question_id = 0

        for split, tasks in data_in_splits.items():
            print(
                f"Starting to query the model with {split.upper()} data...", end="\n\n"
            )
            split_ = Split(name=split)

            for task_id, task in sorted(tasks.items()):
                if (
                    cfg.init_prompt.get("examples", None)
                    and cfg.init_prompt.examples.add
                ):
                    setting.init_prompt.add_examples(
                        task_id=task_id, example_config=cfg.init_prompt.examples
                    )
                else:
                    setting.init_prompt.use_original_prompt()

                task_result = setting.iterate_task(
                    task_id=task_id,
                    task_data=task,
                    prompt_name=f"'{prompt_name}' {prompt_num}/{len(cfg.init_prompt.paths)}",
                )
                split_.add_task(task_result)
                saver.save_task_result(
                    task_id=task_id,
                    task_data=task_result,
                    headers=cfg.results.headers,
                    results_file_name=results_file_names[split],
                    metrics_file_name=metrics_file_names[split],
                    setting=cfg.setting.name,
                )

                print("______________________________", end="\n\n")

            if len(split_.evaluator_after.exact_match_accuracy) != len(tasks):
                raise ValueError(
                    f"Number of tasks and number of accuracies do not match: "
                    f"{len(tasks)} != {len(split_.evaluator_after.exact_match_accuracy)}"
                )

            print(
                f"==> The run for {split.upper()} data is finished successfully <==",
                end="\n\n",
            )

            if multi_system:
                print("The features before applying the setting:")
                print(split_.features_before, end="\n\n")

            print("The features after applying the setting:")
            print(split_.features_after, end="\n\n")

            if multi_system:
                print("Before the setting was applied:")
                split_.evaluator_before.print_accuracies(id_=split)
            split_.evaluator_after.print_accuracies(id_=split)

            if multi_system:
                prompt_evaluator_before.update(split_.evaluator_before)
            prompt_evaluator_after.update(split_.evaluator_after)

            if multi_system:
                saver.save_split_accuracy(
                    evaluator=split_.evaluator_before,
                    accuracy_file_name=metrics_file_names[split],
                )
                saver.save_split_metrics(
                    features=split_.features_before,
                    result_file_names=[
                        metrics_file_names[split],
                        results_file_names[split],
                    ],
                )
                # Plot the prompt accuracies for the split
                plotter.plot_acc_per_task_and_prompt(
                    acc_per_prompt_task={
                        "exact_match_accuracy": split_.evaluator_before.exact_match_accuracy,
                        "soft_match_accuracy": split_.evaluator_before.soft_match_accuracy,
                        "exact_match_std": split_.evaluator_before.exact_match_std,
                        "soft_match_std": split_.evaluator_before.soft_match_std,
                    },
                    y_label="Accuracies and Standard Deviations",
                    plot_name_add=f"{prompt_name}_{split}_before_",
                )
                run_evaluators["before"][split_][init_prompt] = split_.evaluator_before
                run_em_accuracies["before"][split_][
                    init_prompt
                ] = split_.evaluator_before.exact_match_accuracy
                run_sm_accuracies["before"][split_][
                    init_prompt
                ] = split_.evaluator_before.soft_match_accuracy
                run_em_std["before"][split_][
                    init_prompt
                ] = split_.evaluator_before.exact_match_std
                run_sm_std["before"][split_][
                    init_prompt
                ] = split_.evaluator_before.soft_match_std

            saver.save_split_accuracy(
                evaluator=split_.evaluator_after,
                accuracy_file_name=metrics_file_names[split],
            )
            saver.save_split_metrics(
                features=split_.features_after,
                result_file_names=[
                    metrics_file_names[split],
                    results_file_names[split],
                ],
            )
            # Plot the prompt accuracies for the split
            plotter.plot_acc_per_task_and_prompt(
                acc_per_prompt_task={
                    "exact_match_accuracy": split_.evaluator_after.exact_match_accuracy,
                    "soft_match_accuracy": split_.evaluator_after.soft_match_accuracy,
                    "exact_match_std": split_.evaluator_after.exact_match_std,
                    "soft_match_std": split_.evaluator_after.soft_match_std,
                },
                y_label="Accuracies and Standard Deviations",
                plot_name_add=f"{prompt_name}_{split}_after_",
            )

            run_evaluators["after"][split_][init_prompt] = split_.evaluator_after
            run_em_accuracies["after"][split_][
                init_prompt
            ] = split_.evaluator_before.exact_match_accuracy
            run_sm_accuracies["after"][split_][
                init_prompt
            ] = split_.evaluator_after.soft_match_accuracy
            run_em_std["after"][split_][
                init_prompt
            ] = split_.evaluator_after.exact_match_std
            run_sm_std["after"][split_][
                init_prompt
            ] = split_.evaluator_after.soft_match_std

        print("\n- RUN RESULTS -", end="\n\n")

        prompt_evaluator_before.print_accuracies(id_=init_prompt.name)
        prompt_evaluator_after.print_accuracies(id_=init_prompt.name)

        print(
            "Processed",
            loader.number_of_tasks,
            "tasks in total with",
            loader.samples_per_task,
            "samples in each",
        )
        print(
            "Total samples processed",
            loader.number_of_tasks * loader.samples_per_task,
            end="\n\n",
        )

        if cfg.logging.print_to_file:
            # console printing must be returned
            # if printing was redirected to logs created at the beginning of the script
            # 'log_file' will exist in that case as well
            saver.return_console_printing(log_file)

        print("The run is finished successfully")
        print("______________________________")

    if len(cfg.init_prompt.paths) > 1:
        for split in data_in_splits.keys():
            if multi_system:
                # save and plot the accuracies for the same split of all prompts
                saver.save_run_accuracy(
                    task_ids=loader.tasks,
                    split_evaluators=run_evaluators["before"][split],
                    features=split.features_before,
                    split_name=split.name,
                    after=False,
                )
                plotter.result_path = saver.run_path
                plotter.plot_accuracies(
                    exact_match_accuracies=run_em_accuracies["before"][split],
                    soft_match_accuracies=run_sm_accuracies["before"][split],
                    additional_info=f"{split}_before_",
                    compare_prompts=True,
                    label="Accuracy",
                )
                plotter.plot_accuracies(
                    exact_match_accuracies=run_em_std["before"][split],
                    soft_match_accuracies=run_sm_std["before"][split],
                    additional_info=f"{split}_before_",
                    compare_prompts=True,
                    label="Standard Deviation",
                )
            saver.save_run_accuracy(
                task_ids=loader.tasks,
                split_evaluators=run_evaluators["after"][split],
                features=split.features_after,
                split_name=split.name,
                after=True,
            )
            plotter.result_path = saver.run_path
            plotter.plot_accuracies(
                exact_match_accuracies=run_em_accuracies["after"][split],
                soft_match_accuracies=run_sm_accuracies["after"][split],
                additional_info=f"{split}_after_",
                compare_prompts=True,
                label="Accuracy",
            )
            plotter.plot_accuracies(
                exact_match_accuracies=run_em_std["after"][split],
                soft_match_accuracies=run_sm_std["after"][split],
                additional_info=f"{split}_after_",
                compare_prompts=True,
                label="Standard Deviation",
            )

    print("Plots are saved successfully and general accuracies are saved", end="\n\n")

    print("The script has finished running successfully")


if __name__ == "__main__":

    run_setting()
