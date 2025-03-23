from __future__ import annotations

import re
from pathlib import Path

from data.DataLoader import DataLoader
from data.DataSaver import DataSaver
from inference.DataLevels import Task, Sample, Split, SamplePart, Results
from plots.Plotter import Plotter


def mean_attn_score(attn_scores: list[float]) -> float:
    """
    Calculate the mean attention score.

    :param attn_scores: The attention scores.
    :return: The mean attention score.
    """
    return sum(attn_scores) / len(attn_scores)


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


def run(data_path: str, headers: dict[str, list[str]], save_path: str) -> None:
    """
    Run the evaluation pipeline.

    :param data_path: Path to the data for evaluation
    :param headers: The headers for the data
    :param save_path: Path to save the results
    :return: None
    """
    loader = DataLoader()
    saver = DataSaver(save_to=save_path)
    plotter = Plotter(results_path=saver.run_path)

    data_split = extract_split(data_path)

    headers_results_before = [f"{result}_before" for result in headers["results"]]
    headers_results_after = [f"{result}_after" for result in headers["results"]]
    all_headers = headers["general"] + headers_results_before + headers_results_after

    data = loader.load_result_data(
        result_file_path=data_path, headers=all_headers, list_output=True
    )
    before = True if "model_output_before" in data[0].keys() else False

    for row in data:
        remove_unnecessary_columns(row)

    if before:
        # TODO make a function for each repetition
        pass

    interpretability_path = (
        Path(data_path).parent / "after" / "interpretability" / "attn_scores"
    )
    attn_plots_path = interpretability_path / "plots"
    generate_heat_maps = (
        False
        if attn_plots_path.exists() and not any(Path(attn_plots_path).iterdir())
        else True
    )

    sample = None
    task = None
    split = Split(name=data_split, multi_system=False)
    h_patt = re.compile(r"(.+)_(?:after|before)")

    print("General headers:")
    print(headers["general"])

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
            multi_system=False,
        )
        interpretability_result = loader.load_interpretability(
            task_id=part.task_id,
            sample_id=part.sample_id,
            part_id=part.part_id,
            attn_scores_path=str(interpretability_path),
        )
        if generate_heat_maps:
            plotter.draw_heat(
                x=interpretability_result.x_tokens,
                y=interpretability_result.y_tokens,
                scores=interpretability_result.attn_scores,
                task_id=part.task_id,
                sample_id=part.sample_id,
                part_id=part.part_id,
            )
        print(
            "\n".join(
                f"{h_patt.match(result)[1]}: '{row[result]}'"
                for result in headers_results_after
            ),
            sep="\n",
            end="\n\n",
        )
        part.result_after = Results(
            **dict(
                [
                    (
                        (h_patt.match(result)[1], str(row[result]))
                        if h_patt.match(result)
                        else (result, row[result])
                    )
                    for result in headers_results_after
                ]
            ),
            interpretability=interpretability_result,
            after=True,
        )

        if part.sample_id == 1 and part.part_id == 1:
            task = Task(part.task_id, multi_system=False)

        if part.part_id == 1:
            sample = Sample(
                task_id=part.task_id,
                sample_id=part.sample_id,
                multi_system=False,
            )

        sample.add_golden_answers(part.golden_answer)
        sample.add_silver_reasoning(part.silver_reasoning)
        sample.add_part(part)

        if sample and (
            inx == len(data) - 1 or part.sample_id != int(data[inx + 1]["sample_id"])
        ):
            sample.print_sample_predictions()
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
            task.set_results()
            task.evaluator_after.print_accuracies(id_=part.task_id)
            split.add_task(task)

    print("----------------------------------------")
    split.evaluator_after.print_accuracies(id_=data_split)

    if not before:
        plotter.plot_acc_per_task_and_prompt(
            acc_per_prompt_task={
                "exact_match_accuracy_before": split.evaluator_before.exact_match_accuracy,
                "soft_match_accuracy_before": split.evaluator_before.soft_match_accuracy,
                "exact_match_std_before": split.evaluator_before.exact_match_std,
                "soft_match_std_before": split.evaluator_before.soft_match_std,
            },
            y_label="Accuracies and Standard Deviations",
            plot_name_add=f"{split.name}_before_",
        )

    plotter.plot_acc_per_task_and_prompt(
        acc_per_prompt_task={
            "exact_match_accuracy_after": split.evaluator_after.exact_match_accuracy,
            "soft_match_accuracy_after": split.evaluator_after.soft_match_accuracy,
            "exact_match_std_after": split.evaluator_after.exact_match_std,
            "soft_match_std_after": split.evaluator_after.soft_match_std,
        },
        y_label="Accuracies and Standard Deviations",
        plot_name_add=f"{split.name}_after_",
    )


if __name__ == "__main__":
    # TODO: consider running standardize_data.py before running this script if there are not part_ids or silver_reasoning
    data_path = "test/test_join/joined_data2/valid_prompt_init_prompt_direct_answer_results_upd.csv"
    # TODO: make sure that the headers are present in the data
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
            "model_output",  # make sure it's not 'model_result'
        ],
    }
    run(data_path=data_path, headers=headers, save_path="test/test_join/joined_data2/")
