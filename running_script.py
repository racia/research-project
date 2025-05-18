from __future__ import annotations

import os
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
from inference.DataLevels import Split, print_metrics
from inference.Prompt import Prompt
from inference.utils import print_metrics_table
from plots.Plotter import Plotter
from settings.Model import Model
from settings.SD.SpeculativeDecoding import SpeculativeDecoding
from settings.baseline.Baseline import Baseline
from settings.feedback.Feedback import Feedback
from settings.skyline.Skyline import Skyline
from settings.utils import set_device


@hydra.main(version_base=None, config_path=None, config_name=None)
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

    sd = [
        "sd",
        "speculativedecoding",
        "speculative_decoding",
    ]

    if cfg.setting.name.lower() in ["baseline", "skyline"]:
        multi_system = False
    elif cfg.setting.name.lower() in sd or cfg.setting.name.lower() == "feedback":
        multi_system = True
    else:
        raise ValueError(
            f"Setting {cfg.setting.name} is not supported. "
            f"Please choose one of the following: Baseline, Skyline, Feedback, SD or SpeculativeDecoding"
        )

    loader = DataLoader(
        samples_per_task=cfg.data.samples_per_task,
        to_enumerate=cfg.data.to_enumerate,
        wrapper=cfg.data.wrapper,
    )

    data_splits = [split for split, to_use in cfg.data.splits.items() if to_use]
    parts_per_split = {}

    for split in data_splits:
        if cfg.data.get("baseline_results", None):
            parts_per_split[split], _ = loader.load_results(
                results_path=cfg.data.baseline_results,
                data_path=cfg.data.path,
                split=split,
                tasks=cfg.data.task_ids,
                as_parts=True,
            )
        else:
            parts_per_split[split] = loader.load_task_data(
                path=cfg.data.path,
                split=split,
                tasks=cfg.data.task_ids,
                multi_system=multi_system,
            )

    saver = DataSaver(
        save_to=HydraConfig.get().run.dir,
        loaded_baseline_results=bool(cfg.data.baseline_results),
    )
    print(f"Results will be saved to: {saver.results_path}")
    plotter = Plotter(results_path=saver.results_path)

    run_splits = defaultdict(dict)

    if not hasattr(cfg.setting, "name"):
        raise ValueError("The setting name is not provided in the config file")

    if hasattr(cfg, "model"):
        model = Model(
            **cfg.model,
            wrapper=cfg.data.wrapper,
            role=None,
        )
    elif hasattr(cfg, "student"):
        model = Model(
            **cfg.student,
            wrapper=cfg.data.wrapper,
            role="student",
        )
    else:
        raise ValueError("No base model is provided in the config.")

    if cfg.setting.name.lower() == "baseline":
        setting = Baseline(
            model=model,
            total_tasks=loader.number_of_tasks,
            total_parts=loader.number_of_parts,
            samples_per_task=loader.samples_per_task,
            init_prompt=None,
            saver=saver,
        )
    elif cfg.setting.name.lower() == "skyline":
        setting = Skyline(
            model=model,
            total_tasks=loader.number_of_tasks,
            total_parts=loader.number_of_parts,
            samples_per_task=loader.samples_per_task,
            init_prompt=None,
            saver=saver,
        )
    elif cfg.setting.name.lower() == "feedback":
        teacher = Model(**cfg.teacher, role="teacher")
        feedback_prompt = Prompt(
            prompt_path=cfg.feedback_prompt.paths[0],
            wrapper=cfg.feedback_prompt.get("wrapper", None),
            history=cfg.feedback_prompt.get("history", None),
            tokenizer=teacher.tokenizer,
        )
        refine_prompt = Prompt(
            prompt_path=cfg.refine_prompt.paths[0],
            wrapper=cfg.refine_prompt.get("wrapper", None),
            tokenizer=teacher.tokenizer,
        )
        print("- THE FEEDBACK PROMPT -")
        print("______________________________")
        print(feedback_prompt.text)
        print("______________________________", end="\n\n")

        print("- THE REFINE PROMPT -")
        print("______________________________")
        print(refine_prompt.text)
        print("______________________________", end="\n\n", flush=True)

        setting = Feedback(
            student=model,
            teacher=teacher,
            total_tasks=loader.number_of_tasks,
            total_parts=loader.number_of_parts,
            init_prompt=None,
            feedback_prompt=feedback_prompt,
            refine_prompt=refine_prompt,
            teacher_max_new_tokens=cfg.teacher.max_new_tokens,
            student_max_new_tokens=cfg.student.max_new_tokens,
            samples_per_task=cfg.data.samples_per_task,
            saver=saver,
        )
    elif cfg.setting.name.lower() in sd:
        teacher = Model(**cfg.teacher, role="teacher")
        eval_prompt = Prompt(
            prompt_path=cfg.eval_prompt.paths[0],
            history=cfg.eval_prompt.get("history", None),
            wrapper=cfg.eval_prompt.get("wrapper", None),
            tokenizer=teacher.tokenizer,
        )
        # Doesn't need a tokenizer as it still needs to be formatted and
        # will be added to the chat as a message
        resume_prompt = Prompt(
            prompt_path=cfg.resume_prompt.paths[0],
            wrapper=cfg.resume_prompt.get("wrapper", None),
            tokenizer=teacher.tokenizer,
        )
        print("- THE EVAL PROMPT -")
        print("______________________________")
        print(eval_prompt.text)
        print("______________________________", end="\n\n", flush=True)
        print("- THE RESUME PROMPT -")
        print("______________________________")
        print(resume_prompt.text)
        print("______________________________", end="\n\n", flush=True)

        setting = SpeculativeDecoding(
            student=model,
            teacher=teacher,
            total_tasks=loader.number_of_tasks,
            total_parts=loader.number_of_parts,
            init_prompt=None,
            eval_prompt=eval_prompt,
            resume_prompt=resume_prompt,
            samples_per_task=cfg.data.samples_per_task,
            saver=saver,
        )
    else:
        setting = None

    for prompt_num, prompt_path in enumerate(cfg.init_prompt.paths, 1):
        prompt_name = Path(prompt_path).stem
        prompt_evaluator_before = MetricEvaluator(level="prompt", version="before")
        if multi_system:
            prompt_evaluators = [
                prompt_evaluator_before,
                MetricEvaluator(level="prompt", version="after"),
            ]
        else:
            prompt_evaluators = [prompt_evaluator_before]

        prompt_evaluator_after = MetricEvaluator(level="prompt")
        prompt_evaluators = []
        if multi_system:
            prompt_evaluator_before = (
                MetricEvaluator(level="prompt")
            )
            prompt_evaluators.append(prompt_evaluator_before)
        prompt_evaluators.append(prompt_evaluator_after)
        
        log_file_name, results_file_names, metrics_file_names = (
            saver.create_result_paths(prompt_name=prompt_name, splits=data_splits)
        )

        # update result paths
        plotter.results_path = saver.results_path

        # Once the printing is redirected to the log file,
        # the system output will be saved there without additional actions
        if cfg.logging.print_to_file:
            # Print the prompt data to the output file
            # so that it was clear which prompt was used last
            print(f"Starting to query with the prompt: {prompt_name}")
            print(f"Prompt path: {prompt_path}", end="\n\n")
            print(
                f"Redirecting the system output to: {saver.results_path / log_file_name}",
                flush=True,
            )
            log_file = saver.redirect_printing_to_log_file(log_file_name)

            # Print the config data to the log file
            if multi_system:
                print(f"Using the student model: {cfg.student.name}")
                print(f"Using the teacher model: {cfg.teacher.name}")
            else:
                print(f"Using the model: {cfg.model.name}")

        init_prompt = Prompt(
            prompt_path=prompt_path, name=prompt_name, tokenizer=model.tokenizer
        )
        setting.init_prompt = init_prompt

        print(f"Prompt: {prompt_name}, path: {prompt_path}")

        print("- THE SYSTEM SPROMPT -")
        print("______________________________")
        print(init_prompt.text)
        print("______________________________", end="\n\n")

        for split_name, tasks in parts_per_split.items():
            print(
                f"Starting to query the model with {split_name.upper()} data...",
                end="\n\n",
            )
            split = Split(name=split_name, multi_system=multi_system)
            setting.question_id = 0

            for task_id, task in tasks.items():
                if (
                    cfg.init_prompt.get("examples", None)
                    and cfg.init_prompt.examples.add
                ):
                    assert type(task_id) is int
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
                split.add_task(task_result)
                saver.save_task_result(
                    task_id=task_id,
                    task_data=task_result,
                    headers=cfg.results.headers,
                    results_file_name=results_file_names[split.name],
                    metrics_file_name=metrics_file_names[split.name],
                )

                print("______________________________", end="\n\n")

            number_of_accuracies = [
                len(e.exact_match_accuracy) for e in split.evaluators
            ]
            if len(tasks) not in number_of_accuracies:
                raise ValueError(
                    f"Number of tasks and number of accuracies do not match: "
                    f"{len(tasks)} not in {number_of_accuracies}"
                )

            print(
                f"==> The run for {split.name.upper()} data is finished successfully <==",
                end="\n\n",
            )
            items = zip(
                split.versions, split.features, split.evaluators, prompt_evaluators
            )
            for version, features, split_eval, prompt_eval in items:
                print(f"The features {version} applying the setting:")
                print(features, end="\n\n")
                saver.save_split_features(
                    features=features,
                    metrics_file_name=metrics_file_names[split.name],
                    version=version,
                )
                prompt_eval.update(split_eval)
                # Plot the prompt metrics for the split
                # plotter.plot_acc_per_task_and_prompt(
                #     acc_per_prompt_task=split_eval.get_accuracies(as_lists=True),
                #     y_label="Accuracies and Standard Deviations",
                #     plot_name_add=[prompt_name, split.name, version],
                # )

            print_metrics(split)
            saver.save_split_metrics(
                split=split,
                metric_file_name=metrics_file_names[split.name],
            )
            run_splits[split.name][init_prompt] = split

            # if multi_system:
            #     plotter.plot_acc_per_task_and_prompt(
            #         acc_per_prompt_task={
            #             **split.evaluators[0].get_accuracies(as_lists=True),
            #             **split.evaluators[1].get_accuracies(as_lists=True),
            #         },
            #         y_label="Accuracies and Standard Deviations",
            #         plot_name_add=[prompt_name, split.name, "before", "after"],
            #     )

        print("\n- RUN RESULTS -", end="\n\n")

        [prompt_eval.calculate_std() for prompt_eval in prompt_evaluators]
        print_metrics_table(evaluators=prompt_evaluators, id_=init_prompt.name)

        print(
            "Processed",
            loader.number_of_tasks,
            "tasks in total with",
            loader.samples_per_task,
            "samples in each",
        )
        print("Total runs:", setting.total_parts, end="\n\n")

        if cfg.logging.print_to_file:
            # console printing must be returned
            # if printing was redirected to logs created at the beginning of the script
            # 'log_file' will exist in that case as well
            saver.return_console_printing(log_file)

        print("The run is finished successfully")
        print("______________________________")

    if len(cfg.init_prompt.paths) > 1:
        for split_name, prompts_splits in run_splits.items():
            plotter.results_path = saver.run_path
            saver.save_run_accuracy(
                task_ids=loader.tasks,
                splits=prompts_splits,
                split_name=split_name,
            )

    print("The script has finished running successfully")


if __name__ == "__main__":
    run_setting()
