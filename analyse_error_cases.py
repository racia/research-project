# This is a script to analyze error cases of the evaluation results.
# It is designed to be run on the arbitrary number of error cases.
# It defines which tasks or differently filtered cases either are
# always correct or always wrong, which tells us about the complexity of the task
# and allows us to compare the results of different settings.
from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

from evaluation.utils import (
    ERROR_CASES,
    CASES_TO_SIMPLE_ANS,
    CASES_TO_SIMPLE_REAS,
)
from inference.utils import flatten
from plots.Plotter import Plotter
from plots.utils import get_paths


def parse_error_cases(file_path: str) -> list[tuple]:
    """
    Parse the error cases from a given file.
    :param file_path: path to the file containing error cases
    :return: dictionary with error cases categorized by their types
    """
    with open(file_path, "r", encoding="utf-8") as f:
        # drop the first column with the strike-through ID (might be different in different runs)
        # skip the header line (task_id, sample_id, part_id)
        indices = [tuple(map(int, line.split("\t")[1:])) for line in f.readlines()[1:]]
    return indices


def identify_error_cases(path: str) -> dict:
    """
    Identify error cases from the evaluation results at the given path.
    :param path: path to the evaluation results
    :return: dictionary with indices for model answers categorized by cases
    """
    # can have a disambiguation problem with correlation results
    cases_paths = flatten(get_paths(path, "corr", "txt"))
    for case_path in cases_paths:
        matches = {
            case: case_path.match(f"*/{case}.txt") for case in ERROR_CASES.keys()
        }
        if not any(matches):
            print(f"Warning: No error cases found in {case_path}")
            continue
        match_index = [i for i, m in enumerate(matches.values()) if m][0]
        matched_case = list(matches.keys())[match_index]
        ERROR_CASES[matched_case] = parse_error_cases(case_path)

    return ERROR_CASES


def save_indices(
    indices: list | set | dict,
    name,
    results_path: str | Path,
    header: str = "task_id\tsample_id\tpart_id\n",
):
    """
    Save the indices to a file.
    :param indices: list or set of indices to save
    :param case: error case name
    :param setting: setting name
    :param results_path: path to the directory to save the file
    :param header: header for the file, defaults to "task_id\tsample_id\tpart_id\n"
    """
    results_path = Path(results_path) if isinstance(results_path, str) else results_path
    results_path.mkdir(parents=True, exist_ok=True)
    # save the indices to a file
    with open(results_path / f"{name}.txt", "w", encoding="utf-8") as f:
        f.write(header)
        if type(indices) is dict:
            indices = indices.items()
        for inx in indices:
            f.write("\t".join(map(str, inx)) + "\n")


def analyse(**kwargs):
    """
    Analyze the error cases based on the provided keyword arguments.
    :param kwargs: setting-path pairs to the error cases
    """
    # Placeholder for analysis logic
    print("Analyzing with parameters:", kwargs)
    results_path = kwargs.pop("results_path", "reasoning-project/outputs/error_cases")

    print("Results will be saved to:", results_path)
    plotter = Plotter(Path(results_path), color_map="tab20")

    # setting_cases = { setting : { error_case : indices } }
    setting_cases = {
        setting: identify_error_cases(results_path)
        for setting, results_path in kwargs.items()
    }
    print("Identified error cases:")
    print(setting_cases, end="\n\n")

    # NB: outputs with null answer/reasoning are not considered for analysis
    all_identifiers = set()
    # simple cases will have intersecting sets
    simple_cases_settings = {  # defaultdict -> identifier: list of settings
        "ans_corr": defaultdict(list),
        "ans_incorr": defaultdict(list),
        "reas_corr": defaultdict(list),
        "reas_incorr": defaultdict(list),
    }
    cases_settings = {  # defaultdict -> identifier: list of settings
        "ans_corr_reas_corr": defaultdict(list),
        "ans_incorr_reas_corr": defaultdict(list),
        "ans_corr_reas_incorr": defaultdict(list),
        "ans_incorr_reas_incorr": defaultdict(list),
    }
    # simple cases that are allways correct/incorrect across all settings
    simple_unique_cases = {
        "ans_corr": set(),
        "ans_incorr": set(),
        "reas_corr": set(),
        "reas_incorr": set(),
    }
    # full cases that are always correct/incorrect across all settings
    unique_cases = {
        "ans_corr_reas_corr": set(),
        "ans_incorr_reas_corr": set(),
        "ans_corr_reas_incorr": set(),
        "ans_incorr_reas_incorr": set(),
    }
    # filtered_cases = {setting: defaultdict(set) for setting in setting_cases.keys()}
    for setting, cases in setting_cases.items():
        print(f"Setting: {setting}")
        # TODO: how is it possible if unique cases are not counted yet?
        plotter.plot_case_pie(cases, setting, unique=False)
        for level in ("task", "sample", "part"):
            print(f"Plotting histogram grouped by {level}..")
            plotter.plot_error_histogram(
                cases, group_by=level, normalize=False, setting=setting
            )
            plotter.plot_error_histogram(
                cases, group_by=level, normalize=True, setting=setting
            )
        for case, indices in cases.items():
            if not indices:
                warnings.warn(f"Case '{case}' has no indices, skipping..")
                continue
            filtered_indices = set(indices)
            all_identifiers.update(filtered_indices)
            # collect indices and settings for simplified cases (intersecting sets)
            simple_ans = CASES_TO_SIMPLE_ANS[case]
            simple_reas = CASES_TO_SIMPLE_REAS[case]
            # TODO: do we need to not update if there are no data points for a case?
            if filtered_indices:
                for simple in [simple_ans, simple_reas]:
                    if not simple:
                        continue
                    if simple_unique_cases[simple]:
                        simple_unique_cases[simple].intersection_update(
                            filtered_indices
                        )
                    else:
                        simple_unique_cases[simple] = filtered_indices
                    for inx in filtered_indices:
                        simple_cases_settings[simple][inx].append(setting)

            # collect indices and settings for full cases
            if case in unique_cases:
                if unique_cases[case]:
                    unique_cases[case].intersection_update(filtered_indices)
                else:
                    unique_cases[case] = filtered_indices
            if case in cases_settings:
                for inx in filtered_indices:
                    cases_settings[case][inx].append(setting)
            if "null" in case:
                save_indices(indices, f"{case}_{setting}", results_path)

    # TODO: shall I also save the indices of simple cases?

    # saved indices with always correct/incorrect answers/reasonings
    for case, indices in unique_cases.items():
        save_indices(indices, f"always_{case}", results_path)

    # saved indices with filtered always correct/incorrect answers/reasonings
    for setting, cases in setting_cases.items():
        for case, indices in cases.items():
            filtered_indices = set(indices)
            if case in unique_cases:
                filtered_indices -= unique_cases[case]
            if filtered_indices:
                save_indices(filtered_indices, f"filtered_{case}", results_path)

    for case, indices in unique_cases.items():
        tasks_num_of_cases = defaultdict(int)
        for inx in indices:
            task_id = inx[0]
            tasks_num_of_cases[task_id] += 1
        # order by the number of cases starting from the highest
        ordered = dict(
            sorted(tasks_num_of_cases.items(), key=lambda item: item[1], reverse=True)
        )
        save_indices(
            ordered, f"task_order_{case}", results_path, header="task_id\tinstances\n"
        )

    print("Plotting pie charts of always correct/incorrect answers/reasonings..")
    plotter.plot_case_pie(simple_unique_cases, unique=True)
    plotter.plot_case_pie(unique_cases, unique=True)
    print("Plotting histograms of always correct/incorrect answers/reasonings..")
    # important to see the absolute numbers
    plotter.plot_error_histogram(simple_unique_cases, group_by="task", normalize=False)
    plotter.plot_error_histogram(simple_unique_cases, group_by="task", normalize=True)
    plotter.plot_error_histogram(unique_cases, group_by="task", normalize=False)
    plotter.plot_error_histogram(unique_cases, group_by="task", normalize=True)
    print("Plotting heatmaps for always correct/incorrect answers/reasonings..")
    all_cases = {**simple_cases_settings, **cases_settings}
    for case, identifier_settings in all_cases.items():
        if not identifier_settings:
            warnings.warn(
                f"No identifiers for case '{case}', skipping heatmap plotting.."
            )
            continue
        plotter.plot_case_heatmap(
            identifier_settings,
            case_type=case,
            all_indices=all_identifiers,
        )


if __name__ == "__main__":
    # Define the keyword arguments for the analysis function
    # key defines the setting, value defines the path to the evaluation results
    kwargs = {
        # "baseline_reasoning": "/pfs/work9/workspace/scratch/hd_nc326-research-project/baseline/test-eval",
        "baseline_reasoning": "/Users/bohdana.ivakhnenko/PycharmProjects/research-project/outputs/test-eval",
        # TODO: specify the results path if needed (otherwise the plots will be saved to the default path)
        "results_path": "/Users/bohdana.ivakhnenko/PycharmProjects/research-project/outputs/test_error_cases",
    }
    analyse(**kwargs)
