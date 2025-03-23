# Description: This script is used to evaluate the results of the model.
# 1) The script loads the data from the specified path.
# 2) It removes unnecessary columns from the data.
# 3) It extracts the split from the data path.
# 4) It iterates over the data and creates the data levels.
# 5) It loads the silver reasoning and interpretability results.
# 5) It calculates the accuracies for the split and the tasks.
# 6) It plots the accuracies and interpretability heat.
from __future__ import annotations

import re
from pathlib import Path

from data.DataLoader import DataLoader
from data.DataSaver import DataSaver
from inference.DataLevels import Task, Sample, Split, SamplePart, Results
from plots.Plotter import Plotter


def remove_unnecessary_columns(row: dict) -> None:
    """
    Remove unnecessary columns from the row.

    :param row: The row to remove the columns from.
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


def run(
    data_path: str, headers: dict[str, list[str]], save_path: str, multi_system: bool
) -> None:
    """
    Run the evaluation pipeline.

    :param data_path: Path to the data for evaluation
    :param headers: The headers for the data
    :param save_path: Path to save the results
    :param multi_system: Whether the data contains results for two-model setting
    :return: None
    """
    print("You are running the evaluation pipeline.", end="\n\n")
    loader = DataLoader()
    saver = DataSaver(save_to=save_path)
    plotter = Plotter(results_path=saver.run_path)

    data_split = extract_split(data_path)

    interpretability_paths = {}
    generate_heat_maps = {}
    headers_results = {}
    versions = ["before", "after"] if multi_system else ["after"]
    print("The following versions of results will be evaluated:", versions, end="\n\n")

    for version in versions:
        headers_results[version] = [
            f"{result}_{version}" for result in headers["results"]
        ]
        interpretability_paths[version] = (
            Path(data_path).parent / version / "interpretability" / "attn_scores"
        )
        attn_plots_path = interpretability_paths[version] / "plots"
        generate_heat_maps[version] = (
            False
            if attn_plots_path.exists() and not any(Path(attn_plots_path).iterdir())
            else True
        )
        if generate_heat_maps[version]:
            print(
                f"Attention heat maps will be generated for '{version}' results.",
                end="\n\n",
            )

    print("Loading data...", end="\n\n")
    all_headers = headers["general"] + list(headers_results.values())[0]
    data = loader.load_result_data(result_file_path=data_path, list_output=True)
    for row in data:
        remove_unnecessary_columns(row)

    sample = None
    task = None
    split = Split(name=data_split, multi_system=multi_system)
    h_patt = re.compile(r"(.+)_(?:after|before)")
    interpretability_results = {}

    print("\nPART VALUES:\n")

    for inx, row in enumerate(data):
        print(
            "\n".join(
                (
                    f"{gen_header}: '{row[gen_header]}'"
                    if type(row[gen_header]) is str
                    else f"{gen_header}: {row[gen_header]}"
                )
                for gen_header in headers["general"]
            ),
            end="\n",
        )
        part = SamplePart(
            **dict(
                [
                    (
                        (h_patt.match(gen_header)[1], row[h_patt.match(gen_header)[1]])
                        if h_patt.match(gen_header)
                        else (gen_header, row[gen_header])
                    )
                    for gen_header in headers["general"]
                ]
            ),
            multi_system=multi_system,
        )
        for version in versions:
            interpretability_results[version] = loader.load_interpretability(
                task_id=part.task_id,
                sample_id=part.sample_id,
                part_id=part.part_id,
                attn_scores_path=str(interpretability_paths[version]),
            )
            if generate_heat_maps[version]:
                plotter.draw_heat(
                    x=interpretability_results[version].x_tokens,
                    y=interpretability_results[version].y_tokens,
                    x_label=f"Model Output Tokens ({version})",
                    scores=interpretability_results[version].attn_scores,
                    task_id=part.task_id,
                    sample_id=part.sample_id,
                    part_id=part.part_id,
                )
            print(
                "\n".join(
                    f"{h_patt.match(result)[1]}: '{row[result]}'"
                    for result in headers_results[version]
                ),
                sep="\n",
                end="\n\n",
            )
        if multi_system:
            part.result_before = Results(
                **dict(
                    [
                        (
                            (h_patt.match(result)[1], str(row[result]))
                            if h_patt.match(result)
                            else (result, row[result])
                        )
                        for result in headers_results["before"]
                    ]
                ),
                interpretability=interpretability_results["before"],
                after=False,
            )
        part.result_after = Results(
            **dict(
                [
                    (
                        (h_patt.match(result)[1], str(row[result]))
                        if h_patt.match(result)
                        else (result, row[result])
                    )
                    for result in headers_results["after"]
                ]
            ),
            interpretability=interpretability_results["after"],
            after=True,
        )

        if part.sample_id == 1 and part.part_id == 1:
            task = Task(part.task_id, multi_system=multi_system)

        if part.part_id == 1:
            sample = Sample(
                task_id=part.task_id,
                sample_id=part.sample_id,
                multi_system=multi_system,
            )

        sample.add_golden_answers(part.golden_answer)
        sample.add_silver_reasoning(part.silver_reasoning)
        sample.add_part(part)

        if sample and (
            inx == len(data) - 1 or part.sample_id != int(data[inx + 1]["sample_id"])
        ):
            sample.print_sample_predictions()
            if multi_system:
                exact_match_acc, soft_match_acc = (
                    sample.evaluator_before.calculate_accuracies()
                )
                sample.evaluator_before.print_accuracies(
                    id_=part.sample_id,
                    exact_match_acc=exact_match_acc,
                    soft_match_acc=soft_match_acc,
                )
            exact_match_acc, soft_match_acc = (
                sample.evaluator_after.calculate_accuracies()
            )
            sample.evaluator_after.print_accuracies(
                id_=part.sample_id,
                exact_match_acc=exact_match_acc,
                soft_match_acc=soft_match_acc,
            )
            task.add_sample(sample)

        if task and (
            inx == len(data) - 1 or part.task_id != int(data[inx + 1]["task_id"])
        ):
            if inx < len(data) - 1 and int(data[inx + 1]["task_id"]) - part.task_id > 1:
                raise ValueError(
                    f"Missing data for task {part.task_id + 1}.\n"
                    f"Please run the running_script.py for this task and add the results to the current data with join_data.py"
                )
            task.set_results()

            task_metrics = {
                "task_id": part.task_id,
                **task.evaluator_after.get_metrics(),
            }

            if multi_system:
                task_metrics.update(**task.evaluator_before.get_metrics())

                task.evaluator_before.calculate_std()
                task.evaluator_before.print_accuracies(id_=part.task_id)

            task.evaluator_after.calculate_std()
            task.evaluator_after.print_accuracies(id_=part.task_id)
            split.add_task(task)

            saver.save_output(
                data=[task_metrics],
                headers=list(task_metrics.keys()),
                file_name="eval_script_metrics.csv",
                path_add="",
            )

    print("----------------------------------------")

    evaluators = (
        [split.evaluator_before, split.evaluator_after]
        if multi_system
        else [split.evaluator_after]
    )
    features = (
        [split.features_before, split.features_after]
        if multi_system
        else [split.features_after]
    )

    for version, evaluator, feature in zip(versions, evaluators, features):

        evaluator.calculate_std()
        evaluator.print_accuracies(id_=data_split)

        saver.save_split_accuracy(
            evaluator=evaluator,
            metrics_file_name="eval_script_metrics.csv",
        )
        saver.save_split_metrics(
            features=feature,
            metrics_file_names=["eval_script_metrics.csv"],
        )

        print(
            f"Plotting accuracies and standard deviation for results '{version}'...",
            end="\n\n",
        )

        plotter.plot_acc_per_task_and_prompt(
            acc_per_prompt_task=evaluator.get_metrics(as_lists=True),
            y_label="Accuracies and Standard Deviations",
            plot_name_add=f"{split.name}; {version}",
        )

    print("\nThe evaluation pipeline has finished successfully.")


if __name__ == "__main__":
    # TODO: consider running standardize_data.py before running this script if there are not part_ids or silver_reasoning
    data_path = "test/test_join/joined_data2/valid_prompt_init_prompt_direct_answer_results_upd.csv"
    # TODO: provide a path to directory to save the standardized data
    save_directory = "test/test_join/joined_data2/"
    # TODO: does the data feature before and after results? If yes, set to True
    multi_system = False
    # TODO: make sure that all the important headers are on the list and present in the data
    headers = {
        "general": [
            "id_",  # TODO: make sure the id_ is present in the data, not 'id'
            "task_id",
            "sample_id",
            "part_id",
            "task",
            "golden_answer",
            "silver_reasoning",
            "correct",
        ],
        "results": [  # for both before and after
            "model_answer",
            "model_reasoning",
            "model_output",  # TODO: make sure it's not 'model_result'
        ],
    }
    run(
        data_path=data_path,
        headers=headers,
        save_path=save_directory,
        multi_system=multi_system,
    )
