# Description: This script is used to evaluate the results of the model.
# 1) The script loads the data from the specified path.
# 2) It removes unnecessary columns from the data.
# 3) It extracts the split from the data path.
# 4) It iterates over the data and creates the data levels.
# 5) It loads the silver reasoning and interpretability results.
# 5) It calculates the metrics for the split and the tasks.
# 6) It plots the metrics and interpretability heat.
# 7) It saves the results to the specified path.
# 8) It prints the metrics table.
# 9) It categorizes the results into different cases.
# 10) It saves the categorized results to the specified path.
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from data.DataLoader import DataLoader
from data.DataSaver import DataSaver
from data.utils import format_metrics
from inference.DataLevels import Results, Sample, Split, Task, print_metrics
from inference.utils import print_metrics_table
from plots.Plotter import Plotter
from evaluation.feedback_anaylsis import FeedbackAnalysis
from evaluation.setting_analysis import SettingAnalysis

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
    results_path: str,
    save_path: str,
    samples_per_task: int,
    experiment: str,
    setting: str = "baseline",
    create_heatmaps: bool = True,
    verbose: bool = False,
) -> None:
    """
    Run the evaluation pipeline.

    :param results_path: Path to the data for evaluation
    :param save_path: Path to save the results
    :param samples_per_task: Number of samples per task the results were ran with
    :param experiment: The experiment to evaluate (e.g., "reasoning_answer", "direct_answer")
    :param setting: The setting of the experiment (e.g., "baseline", "feedback")
    :param verbose: Whether to print the results to the console
    :param create_heatmaps: Whether to create heatmaps for the interpretability results
    :return: None
    """
    print("You are running the evaluation pipeline.", end="\n\n")

    experiment = experiment.lower()
    supported_experiments = ["reasoning_answer", "direct_answer"]
    if experiment not in supported_experiments:
        raise ValueError(
            f"Experiment '{experiment}' is not supported. "
            f"Please choose either of {supported_experiments}"
        )
    setting = setting.lower()
    supported_settings = [
        "baseline",
        "feedback",
        "skyline",
        "speculative_decoding",
        "sd",
    ]
    if setting not in supported_settings:
        raise ValueError(
            f"Setting not recognized, expected one of: {supported_settings}"
        )
    if not results_path:
        raise ValueError("Please provide a path to the data for evaluation.")

    if setting == "feedback":
        feedback_result_path = str(Path(save_path) / "feedback_analysis")
        teacher_dfs = DataLoader.load_iteration_data(results_path, 'teacher', 'feedback')
        student_dfs = DataLoader.load_iteration_data(results_path, 'student', 'feedback')
        # run feedback analysis

        feedback = FeedbackAnalysis(teacher_dfs, student_dfs, feedback_result_path)
        feedback.run_all()

        shared = SettingAnalysis(student_dfs, feedback_result_path, teacher_dfs)
        difficulty_df = shared.analyse_task_difficulty()
        shared.cluster_errors(n_clusters=5)

    print("Loading data...", end="\n\n")
    loader = DataLoader(prefix=PREFIX, samples_per_task=samples_per_task)
    # loaded results in parts with original data, tokens-ids, and interpretability results
    results_data, multi_system = loader.load_results(
        results_path=results_path,
        data_path="../tasks_1-20_v1-2/en-valid/",
        split=extract_split(results_path),
        as_parts=True,
    )
    # maybe loaded_baseline_results is not needed for evaluation
    saver = DataSaver(
        save_to=str(Path(save_path) / "eval"),
        loaded_baseline_results=True if multi_system else False,
    )
    results_file_name = f"{Path(results_path).stem}_upd.csv"
    plotter = Plotter(results_path=saver.run_path, color_map="tab20")

    print(f"\nLoaded results data for {len(results_data)} tasks.")
    print(f"Loaded {loader.number_of_parts} sample parts created from raw data.")

    data_split = extract_split(results_path)
    sample, task, split = None, None, Split(name=data_split, multi_system=multi_system)
    for task_id, samples in results_data.items():
        assert type(task_id) is int
        task = Task(task_id, multi_system=multi_system)

        for sample_id, parts in samples.items():
            assert type(sample_id) is int
            sample = Sample(
                task_id=task_id,
                sample_id=sample_id,
                multi_system=multi_system,
            )
            # Used to store the correct answers for each sample for later evaluation

            for part in parts:
                for version, result in zip(part.versions, part.results):
                    # TODO: add reasoning judgment to part results for it to be saved in the results table

                    if create_heatmaps and not result.interpretability.empty():
                        plotter.draw_heat(
                            result.interpretability,
                            x_label="Sentence Indices",
                            task_id=part.task_id,
                            sample_id=part.sample_id,
                            part_id=part.part_id,
                            title=f"Attention Map for Task {part.task_id} Sample {part.sample_id} "
                            f"Part {part.part_id} (version: {version}, case: {result.category})",
                        )

                sample.add_part(part)
                result = part.get_result()

                # necessary only if we want to addition more columns to our original results
                # otherwise we can just create separate tables or files
                saver.save_output(
                    data=[result],
                    headers=list(result.keys()),
                    file_name=results_file_name,
                )
            sample.calculate_metrics()
            task.add_sample(sample)

            if verbose:
                sample.print_sample_predictions()
                print_metrics(sample)
            for evaluator, version in zip(sample.evaluators, sample.versions):
                metrics = list(
                    format_metrics(evaluator.get_metrics(as_lists=True)).values()
                )
                print(f"Metrics for {evaluator.level} {version}:", metrics, end="\n\n")

        task.set_results()

        split.add_task(task)
        print(
            f"Added task {task_id} with {len(sample.parts)} parts to split {data_split}."
        )

        if verbose:
            print_metrics(task)
        task_corr_matrices = task.calculate_metrics()
        for version, evaluator, corr_matrix in zip(
            task.versions, task.evaluators, task_corr_matrices.values()
        ):
            # Plot Attention vs Seen Context Lengths for the Task
            plotter.plot_correlation(
                    x_data={"seen_context_lengths": task.seen_context_lengths},
                    y_data=evaluator.parts_attn_on_target.all,
                    x_label="Seen Context Lengths",
                    y_label="Attention on Target Tokens",
                    file_name=f"attn_on_target.pdf",
                    plot_name_add=[f"Task-{task_id}"],
                    path_add=Path(version, f"Task-{task_id}"),
                    level="task",
                )
            # Attn on Target for Accuracy
            plotter.plot_correlation(
                x_data=evaluator.get_accuracies(as_lists=True),
                y_data=evaluator.attn_on_target.all,
                x_label="Accuracy",
                y_label="Attention on Target Tokens",
                file_name=f"acc-attn_on_target.pdf",
                plot_name_add=[f"Task-{task_id}"],
                path_add=Path(version, f"Task-{task_id}"),
                level="task",
                include_soft=False,
                label_add=[f"s{sample.sample_id}" for sample in task.samples],
            )

            # Attn on Target for Target Distances by Answer Correct
            plotter.plot_corr_boxplot(
                x_data={
                    "parts_target_distances": task.parts_target_distances.all
                },
                y_data={
                    "parts_attn_on_target": evaluator.parts_attn_on_target.all,
                    "parts_answer_correct": evaluator.parts_answer_correct.all,
                    "parts_features": task.parts_features[version],
                },
                x_label="Target Sentence Distances",
                y_label="Attention On Target",
                displ_percentage=False,
                version=version,
                file_name=f"attn-target_distances.pdf",
                plot_name_add=[f"Task-{task_id}"],
                path_add=Path(version, f"Task-{task_id}"),
            )
            # Attn on Target for Answer Correct by Parts Features
            plotter.plot_corr_boxplot(
                x_data={
                    "parts_answer_correct": evaluator.parts_answer_correct.all,
                },
                y_data={
                    "parts_attn_on_target": evaluator.parts_attn_on_target,
                    "parts_features": task.parts_features[version],
                },
                x_label="Answer Correct",
                y_label="Attention On Target",
                displ_percentage=False,
                version=version,
                file_name=f"attn-ans_correct.pdf",
                plot_name_add=[f"Task-{task_id}"],
                path_add=Path(version, f"Task-{task_id}"),
            )

            # Attn on target for Anwer in Self by Answer Correct
            plotter.plot_corr_boxplot(
                x_data={
                    "parts_answer_in_self": task.parts_answer_in_self
                },
                y_data={
                    "parts_attn_on_target": evaluator.parts_attn_on_target.all,
                    "parts_answer_correct": evaluator.parts_answer_correct.all,
                },
                x_label="Answer In Self",
                y_label="Attention On Target",
                displ_percentage=False,
                version=version,
                file_name=f"attn-ans_in_self.pdf",
                plot_name_add=[f"Task-{task_id}"],
                path_add=Path(version, f"Task-{task_id}"),
            )
            # Attn on Target for Seen Context Lengths by Answer Correct
            plotter.plot_corr_boxplot(
                x_data={
                    "parts_seen_context_lengths": task.seen_context_lengths
                },
                y_data={
                    "parts_attn_on_target": evaluator.parts_attn_on_target.all,
                    "parts_answer_correct": evaluator.parts_answer_correct.all,
                },
                x_label="Seen Context Lengths",
                y_label="Attention On Target",
                displ_percentage=False,
                version=version,
                file_name=f"attn-seen_context_lengths.pdf",
                plot_name_add=[f"Task-{task_id}"],
                path_add=Path(version, f"Task-{task_id}"),
            )
            # Answer Correct for Seen Context Lengths by Answer In Self
            plotter.plot_corr_hist(
                x_data={
                    "parts_seen_context_lengths": task.seen_context_lengths
                },
                y_data={
                    "parts_answer_correct": evaluator.parts_answer_correct.all,
                    "parts_answer_in_self": task.parts_answer_in_self,
                },
                x_label="Seen Context Lengths",
                y_label="Parts Answer In[Correct]",
                displ_percentage=True,
                file_name=f"parts_answer_correct.pdf",
                plot_name_add=[f"Task-{task_id}"],
                path_add=Path(version, f"Task-{task_id}"),
            )

            plotter.correlation_map(
                data=corr_matrix,
                level=evaluator.level,
                version=version,
                file_name=f"corr_matrix_task_{task_id}.pdf",
                id=task_id,
            )

            saver.save_json(
                data=corr_matrix,
                file_path=f"corr_matrix_task_{task_id}.json",
                path_add=Path(version),
            )

            metrics_to_save = defaultdict(dict)
            metrics = list(
                format_metrics(evaluator.get_metrics(as_lists=True)).values()
            )
            for metric in metrics:
                metrics_to_save[metric["task_id"]].update(metric)

            for metric in metrics_to_save.values():
                saver.save_output(
                    data=[metric],
                    headers=list(metric.keys()),
                    file_name=f"eval_script_metrics_{version}.csv",
                    path_add=Path(version),
                )

    if verbose:
        print_metrics_table(evaluators=split.evaluators, id_=data_split)

    saver.save_split_metrics(
        split=split,
        metric_file_name="eval_script_metrics.csv",
    )

    split_corr_matrices = split.calculate_metrics()
    for version, evaluator, features, corr_matrix in zip(
        split.versions, split.evaluators, split.features, split_corr_matrices.values()
    ):
        # SAVING
        saver.save_json(
            data=corr_matrix,
            file_path=f"corr_matrix_split_{split.name}.json",
            path_add=version,
        )
        saver.save_split_features(
            features=features,
            metrics_file_name="eval_script_features.csv",
            version=version,
        )

        # PLOTTING
        plotter.correlation_map(
            data=corr_matrix,
            level=evaluator.level,
            version=version,
            split_name=split.name,
            file_name=f"corr_matrix_split_{split.name}.pdf",
        )
        # Plot Accuracy vs Attn on Target for the Split
        plotter.plot_correlation(
                x_data=evaluator.get_accuracies(as_lists=True),
                y_data=evaluator.attn_on_target.all,
                x_label="Accuracy",
                y_label="Attention on Target Tokens",
                file_name=f"acc-attn_on_target_{split.name}.pdf",
                plot_name_add=[f"Split-{split.name}"],
                path_add=Path(version, f"Split-{split.name}"),
                level="split",
                include_soft=False,
                label_add=[f"t{task.task_id}" for task in split.tasks],
            )

        # Attn on Target for Seen Context Lengths by Answer Correct
        plotter.plot_corr_boxplot(
            x_data={"seen_context_lengths": split.seen_context_lengths},
            y_data={
                "parts_attn_on_targets": evaluator.parts_attn_on_target.all,
                "parts_answer_correct": evaluator.parts_answer_correct.all,
            },
            x_label="Seen Context Lengths",
            y_label="Attention On Target",
            displ_percentage=False,
            version=version,
            level="split",
            file_name=f"attn-seen_context_lengths_{split.name}.pdf",
            plot_name_add=[f"Split-{split.name}"],
            path_add=Path(version, f"Split-{split.name}"),
        )
        # Attn on Target for Target Distances by Answer Correct
        plotter.plot_corr_boxplot(
            x_data={"parts_target_distances": split.parts_target_distances},
            y_data={
                "parts_attn_on_targets": evaluator.parts_attn_on_target.all,
                "parts_answer_correct": evaluator.parts_answer_correct.all,
            },
            x_label="Target Sentence Distances",
            y_label="Attention On Target",
            level="split",
            displ_percentage=False,
            version=version,
            file_name=f"attn-target_distances_{split.name}.pdf",
            plot_name_add=[f"Split-{split.name}"],
            path_add=Path(version, f"Split-{split.name}"),
        )
        # Answer Correct for Seen Context Lengths by Answer In Self
        plotter.plot_corr_hist(
            x_data={"parts_seen_context_lengths": split.seen_context_lengths},
            y_data={
                "parts_answer_correct": evaluator.parts_answer_correct.all,
                "parts_answer_in_self": split.parts_answer_in_self,
            },
            x_label="Parts Seen Context Lengths",
            y_label="Parts Answer [In]Correct",
            level="split",
            displ_percentage=True,
            file_name=f"parts_answer_correct_{split.name}.pdf",
            plot_name_add=[f"Split-{split.name}"],
            path_add=Path(version, f"Split-{split.name}"),
        )
        print(
            f"\nPlotting accuracies and standard deviation for results '{version}'...",
            end="\n\n",
        )
        plotter.plot_acc_with_std(
            acc_per_prompt_task=evaluator.get_accuracies(as_lists=True),
            y_label="Accuracies with Standard Deviations",
            plot_name_add=[split.name, version],
        )
        print(
            f"\nPlotting attentions for results '{version}'...",
            end="\n\n",
        )
        plotter.plot_acc_with_std(
            acc_per_prompt_task=evaluator.get_attentions(as_lists=True),
            y_label="Attentions",
            plot_name_add=[split.name, version],
        )
        print(
            f"\nPlotting reasoning scores for results '{version}'...",
            end="\n\n",
        )
        plotter.plot_acc_with_std(
            acc_per_prompt_task=evaluator.get_reasoning_scores(as_lists=True),
            y_label="Reasoning Scores",
            plot_name_add=[split.name, version],
        )
        print(
            f"\nPlotting correlations for results '{version}' between metrics:",
            evaluator.get_correlations(as_lists=True),
            end="\n\n",
        )

        # ERROR CASES
        print("Saving result categories...")
        plotter.plot_answer_type_per_part(
            Results.CASE_COUNTERS[version],
            specification={
                "setting": setting,
                "experiment": experiment,
                "version": version,
            },
        )
        for score in ("bleu", "rouge", "meteor"):
            plotter.plot_answer_type_per_part(
                Results.CASE_COUNTERS[version],
                specification={
                    "setting": setting,
                    "experiment": experiment,
                    "version": version,
                    "score": score.upper(),
                },
                reasoning_scores=evaluator.__getattribute__(f"ids_with_{score}"),
            )
        plotter.plot_answer_type_per_part(
            Results.CASE_COUNTERS[version],
            specification={
                "setting": setting,
                "experiment": experiment,
                "version": version,
                "score": "ATTN_ON_TARGET",
            },
            reasoning_scores=evaluator.ids_with_attn_on_target,
        )
        for case, case_list in Results.CASE_COUNTERS[version].items():
            headers = "id_\ttask_id\tsample_id\tpart_id"
            if case_list:
                saver.save_with_separator(
                    saver.run_path / version / f"{case}.txt",
                    [headers] + case_list,
                    sep="\n",
                )
                print(f"Case {case}: detected {len(case_list)} occurrences.")
            else:
                print(f"Case {case}: detected 0 occurrences. Nothing!")

    print(f"Plots produced: {plotter.plot_counter_prompt}")
    print("\nThe evaluation pipeline has finished successfully.")


def parse_args(script_args: str | list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line arguments.

    :return: None
    """
    parser = argparse.ArgumentParser(description="Evaluate the results of the model.")
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to the data.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path where to save the results.",
    )
    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=50,
        help="Number of samples per task the results were ran with (check your config for the run).",
    )
    # Shouldn't always be True?
    parser.add_argument(
        "--create_heatmaps",
        action="store_true",
        help="Whether to create attention heatmaps for the interpretability results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print the results to the console.",
    )

    parser.add_argument(
        "--setting",
        type=str,
        default="baseline")

    if script_args is not None:
        if isinstance(script_args, str):
            script_args = script_args.split()
        args, unexpected = parser.parse_known_args(
            script_args, namespace=argparse.Namespace()
        )
        if unexpected:
            print(f"Unexpected arguments: {unexpected}")
        return args
    return parser.parse_args()


if __name__ == "__main__":
    # path = "--results_path /pfs/work9/workspace/scratch/hd_nc326-research-project/baseline/test/reasoning/all_tasks/joined_reasoning_results_task_results.csv"
    # args = " --save_path /pfs/work9/workspace/scratch/hd_nc326-research-project/baseline/test-eval/joined-data --samples_per_task 3 --verbose"
    args = parse_args()
    # python3.12 evaluate_data.py --results_path baseline/28-05-2025/22-39-52/init_prompt_reasoning/valid_init_prompt_reasoning_results.csv --save_path results/here --samples_per_task 15 --create_heatmaps --verbose
    run(
        results_path="/pfs/work9/workspace/scratch/hd_nc326-research-project/feedback/test/all_tasks_joined_old/iterations", #args.results_path,
        save_path="/pfs/work9/workspace/scratch/hd_nc326-research-project/feedback/test1",#args.save_path,
        samples_per_task=args.samples_per_task,
        setting=args.setting,
        experiment="reasoning_answer",
        create_heatmaps=args.create_heatmaps,
        verbose=args.verbose,
    )
