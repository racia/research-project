# Description: This script is used to evaluate the results of the model.
# 1) The script loads the data from the specified path.
# 2) It removes unnecessary columns from the data.
# 3) It extracts the split from the data path.
# 4) It iterates over the data and creates the data levels.
# 5) It loads the silver reasoning and interpretability results.
# 5) It calculates the accuracies for the split and the tasks.
# 6) It plots the accuracies and interpretability heat.
# 7) It saves the results to the specified path.
# 8) It prints the metrics table.
# 9) It categorizes the results into different cases.
# 10) It saves the categorized results to the specified path.
from __future__ import annotations

import re
import warnings
from collections import defaultdict
from pathlib import Path

from data.DataLoader import DataLoader, SilverReasoning
from data.DataSaver import DataSaver
from data.utils import structure_parts
from inference.DataLevels import (
    Task,
    Sample,
    Split,
    Results,
    print_metrics,
)
from inference.utils import print_metrics_table
from plots.Plotter import Plotter
from settings.config import DataSplits

PREFIX = Path.cwd()
while PREFIX.name != "research-project":
    PREFIX = PREFIX.parent


def remove_unnecessary_columns(
    row: dict[str, str | int | float], headers: dict[str, list[str]]
) -> None:
    """
    Remove unnecessary columns from the row.

    :param row: The row to remove the columns from.
    :param headers: The headers of the columns.
    :return: None
    """
    unnecessary_columns = [
        "correct",
        "correct?",
        "exact_match_accuracy",
        "soft_match_accuracy",
        "there",
        "verbs",
        "pronouns",
        "not_mentioned",
    ]
    for col in unnecessary_columns:
        if col in row:
            del row[col]
    if not row["silver_reasoning"]:
        del row["silver_reasoning"]
        if "silver_reasoning" in headers["general"]:
            del headers["general"][headers["general"].index("silver_reasoning")]


def extract_split(path) -> str:
    """
    Extract the split from the data path. If the split is not found, return "split".

    :param path: The path to the data.
    :return: The split.
    """
    for split in ["valid", "test", "train"]:
        if split in path:
            return split
    return "split"


def structure_result(headers_results: list[str], row: dict, version) -> dict[str, list]:
    """
    Structure the result into a dictionary.
    :param headers_results: The headers and results.
    :param row: The row of data to structure.
    :return: The structured result.
    """
    h_patt = re.compile(r"(.+)_(?:after|before)")
    result = [
        (
            (h_patt.match(header)[1], str(row[f"{header}_{version}"]))
            if h_patt.match(header)
            else (header, row[f"{header}_{version}"])
        )
        for header in headers_results
    ]
    return dict(result)


def get_result(
    results_data: list[dict], task_id: int, sample_id: int, part_id: int
) -> dict[str, str] | None:
    """
    Get the result for the task_id, sample_id and part_id.
    """
    for row in results_data:
        if (
            row["task_id"] == task_id
            and row["sample_id"] == sample_id
            and row["part_id"] == part_id
        ):
            return row
    return None


def run(
    data_path: str,
    save_path: str,
    samples_per_task: int,
    create_heatmaps: bool = True,
    verbose: bool = False,
) -> None:
    """
    Run the evaluation pipeline.

    :param data_path: Path to the data for evaluation
    :param save_path: Path to save the results
    :param samples_per_task: Number of samples per task the results were ran with
    :param create_heatmaps: Whether to create heatmaps for the interpretability results
    :param verbose: Whether to print the results to the console
    :return: None
    """
    print("You are running the evaluation pipeline.", end="\n\n")

    if not data_path:
        raise ValueError("Please provide a path to the data for evaluation.")

    headers = {
        "general": [
            "id_",
            "task_id",
            "sample_id",
            "part_id",
            "task",
            "golden_answer",
            "silver_reasoning",
        ],
        "results": [  # for both before and after
            "model_answer",
            "model_reasoning",
            "model_output",
        ],
    }
    loader = DataLoader(prefix=PREFIX, samples_per_task=samples_per_task)
    silver_reasoning = SilverReasoning(loader)
    saver = DataSaver(save_to=save_path)
    plotter = Plotter(results_path=saver.run_path)
    results_file_name = f"{Path(data_path).stem}_upd.csv"

    print("Loading data...", end="\n\n")

    multi_system = False
    samples_per_task = defaultdict(list)
    results_data = loader.load_results(results_path=data_path, list_output=True)

    for row in results_data:
        samples_per_task[row["task_id"]].append(row["sample_id"])
        remove_unnecessary_columns(row, headers)
        if "model_answer_before" in row:
            multi_system = True

    for task_id, sample_ids in samples_per_task.items():
        print(f"Found {len(set(sample_ids))} samples for task ID {task_id}.")

    task_ids = sorted(list(set([row["task_id"] for row in results_data])))
    raw_parts = loader.load_task_data(
        path="../tasks_1-20_v1-2/en-valid/",
        split=DataSplits.valid,
        tasks=task_ids,
        multi_system=multi_system,
        flat=True,
    )

    print(f"\nLoaded results data for {len(results_data)} tasks.")
    print(f"Loaded {len(raw_parts)} sample parts created from raw data.")

    data_split = extract_split(data_path)
    sample, task, split = None, None, Split(name=data_split, multi_system=multi_system)
    structured_levels = structure_parts(raw_parts)

    for task_id, samples in structured_levels.items():
        task = Task(task_id, multi_system=multi_system)

        for sample_id, parts in samples.items():
            sample = Sample(
                task_id=task_id,
                sample_id=sample_id,
                multi_system=multi_system,
            )
            for part in parts:
                row = get_result(
                    results_data=results_data,
                    task_id=part.task_id,
                    sample_id=part.sample_id,
                    part_id=part.part_id,
                )

                if not row:
                    warnings.warn(
                        f"Skipping part {part.id_}: task {part.task_id}, "
                        f"sample {part.sample_id}, part {part.part_id} (not found in results data)."
                    )
                    continue

                if (
                    part.sample_id != sample.sample_id
                    or part.sample_id != row["sample_id"]
                ):
                    raise ValueError(
                        f"Sample ID {part.sample_id} does not match with the row sample ID "
                        f"{row['sample_id']}: row {row['id_']}."
                    )

                if part.task_id != task.task_id or part.task_id != row["task_id"]:
                    raise ValueError(
                        f"Task ID {part.task_id} does not match with the task ID "
                        f"{task.task_id}: row {row['id_']}."
                    )

                if not part.silver_reasoning:
                    part.silver_reasoning = silver_reasoning.get(
                        task_id=part.task_id,
                        sample_id=part.sample_id,
                        part_id=part.part_id,
                        split=split.name,
                    )
                sample.add_silver_reasoning(part.silver_reasoning)
                sample.add_golden_answers(part.golden_answer)

                for version, result in zip(part.versions, part.results):
                    interpr_path = (
                        Path(data_path).parent
                        / version
                        / "interpretability"
                        / "attn_scores"
                    )
                    interpretability_result = loader.load_interpretability(
                        task_id=part.task_id,
                        sample_id=part.sample_id,
                        part_id=part.part_id,
                        attn_scores_path=str(interpr_path),
                    )
                    part.set_output(
                        **structure_result(headers["results"], row, version),
                        interpretability=interpretability_result,
                        version=version,
                    )
                    attn_plots_path = interpr_path / "plots"
                    plots_present = attn_plots_path.exists() and not any(
                        Path(attn_plots_path).iterdir()
                    )
                    if not plots_present and create_heatmaps:
                        plotter.draw_heat(
                            x=interpretability_result.x_tokens,
                            y=interpretability_result.y_tokens,
                            x_label=interpretability_result.x_label,
                            scores=interpretability_result.attn_scores,
                            task_id=part.task_id,
                            sample_id=part.sample_id,
                            part_id=part.part_id,
                            title=f"Attention Map for Task {part.task_id} Sample {part.sample_id} "
                            f"Part {part.part_id} ({version}, {result.category})",
                        )

                sample.add_part(part)

                result = part.get_result()
                saver.save_output(
                    data=[result],
                    headers=list(result.keys()),
                    file_name=results_file_name,
                )

            for evaluator in sample.evaluators:
                evaluator.calculate_accuracies()
            task.add_sample(sample)

            if verbose:
                sample.print_sample_predictions()
                print_metrics(sample, table=True)

        task.set_results()
        task_accuracies = {
            "task_id": task.task_id,
        }
        for evaluator in task.evaluators:
            task_accuracies.update(**evaluator.get_accuracies())
            evaluator.calculate_std()

        if verbose:
            print_metrics(task, table=True)

        split.add_task(task)
        saver.save_output(
            data=[task_accuracies],
            headers=list(task_accuracies.keys()),
            file_name="eval_script_accuracies.csv",
            path_add="",
        )

    if verbose:
        print_metrics_table(*split.evaluators, id_=data_split)

    saver.save_split_accuracy(
        evaluators=split.evaluators,
        accuracy_file_name="eval_script_accuracies.csv",
        multi_system=multi_system,
    )

    for version, evaluator, features in zip(
        split.versions, split.evaluators, split.features
    ):
        saver.save_split_metrics(
            features=features,
            metrics_file_name="eval_script_metrics.csv",
            version=version,
        )
        print(
            f"\nPlotting accuracies and standard deviation for results '{version}'...",
            end="\n\n",
        )
        plotter.plot_acc_per_task_and_prompt(
            acc_per_prompt_task=evaluator.get_accuracies(as_lists=True),
            y_label="Accuracies and Standard Deviations",
            plot_name_add=[split.name, version],
        )
        # TODO: plot attention distribution per task and sample
        print("Saving result categories...")
        for case, case_list in Results.CASE_COUNTERS[version].items():
            headers = "id_\ttask_id\tsample_id\tpart_id"
            if case_list:
                saver.save_with_separator(
                    saver.run_path / version / f"{case}.txt",
                    [headers] + case_list,
                    sep="\n",
                )
                print(f"Saved {len(case_list)} cases for {case}.")
            else:
                print(f"No cases for {case}.")

    print("\nThe evaluation pipeline has finished successfully.")


if __name__ == "__main__":
    # TODO: consider running standardize_data.py before running this script if there are not part_ids or silver_reasoning
    data_path = "results/baseline/valid/basic_baseline/baseline_prompt/valid_baseline_prompt_results.csv"
    # TODO: provide a path to directory to save the standardized data
    save_directory = "results/baseline/valid/basic_baseline/baseline_prompt/eval"
    run(
        data_path=data_path,
        save_path=save_directory,
        samples_per_task=50,
        create_heatmaps=False,
        verbose=False,
    )
