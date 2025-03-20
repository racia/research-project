from __future__ import annotations

import os
from pathlib import Path

from evaluation.Metrics import Accuracy, Metric
from inference.DataLevels import Features


def is_empty_file(file_path: Path) -> bool:
    """
    Checks if the file exists and is empty.

    :param file_path: the file path to check
    :return: True if file exists and is empty
             False if file is non-empty
    """
    return os.path.isfile(file_path) and os.path.getsize(file_path) == 0


def prepare_accuracy_headers(prompt_name: str = ""):
    """
    Prepare the headers for the accuracies.

    :param prompt_name: the name of the prompt
    :return: the headers for the accuracies
    """
    prompt_name_ = prompt_name.replace("prompt_", "")
    return {
        "exact_match": f"{prompt_name_}_exact_match_accuracy".strip("_"),
        "soft_match": f"{prompt_name_}_soft_match_accuracy".strip("_"),
    }


def format_split_metrics(
    features: Features, headers: dict, metrics_to_save: dict
) -> dict[str, dict]:
    """
    Format the metrics for the split to save them later.

    :param features: the features of the split
    :param headers: accuracy headers
    :param metrics_to_save: the accuracies to save
    :return: the metrics to save
    """
    metrics = {
        "there": features.there,
        "verbs": features.verbs,
        "pronouns": features.pronouns,
        "not_mentioned": features.not_mentioned,
    }
    for metric, value in metrics.items():
        if metric not in metrics_to_save:
            metrics_to_save[metric] = {"task_id": metric}
        metrics_to_save[metric].update({headers["exact_match"]: value})
    return metrics_to_save


def format_accuracy_metrics(
    exact_match_accuracies: Accuracy,
    soft_match_accuracies: Accuracy,
    exact_match_std: Metric,
    soft_match_std: Metric,
    headers: dict = None,
    accuracies_to_save: dict[str, dict] = None,
) -> dict[str, dict]:
    """
    Format the accuracy metrics for the split to save them later:
    - mean accuracy for all tasks
    - standard deviation for all tasks

    :param accuracies_to_save: the accuracies to save
    :param exact_match_accuracies: the exact-match accuracies
    :param soft_match_accuracies: the soft-match accuracies
    :param exact_match_std: the standard deviation for the exact-match accuracies (per task)
    :param soft_match_std: the standard deviation for the soft-match accuracies (per task)
    :param headers: the headers for the accuracies
    """
    accuracy_metrics = {
        "mean": {
            "task_id": "mean",
            (
                headers["exact_match_acc"] if headers else "exact_match_accuracy"
            ): exact_match_accuracies.get_mean(),
            (
                headers["soft_match_acc"] if headers else "soft_match_accuracy"
            ): soft_match_accuracies.get_mean(),
            (
                headers["exact_match_std"] if headers else "exact_match_std"
            ): exact_match_std.get_mean(),
            (
                headers["soft_match_std"] if headers else "soft_match_std"
            ): soft_match_std.get_mean(),
        },
        "std": {
            "task_id": "std",
            (
                headers["exact_match"] if headers else "exact_match_accuracy"
            ): exact_match_accuracies.get_std(),
            (
                headers["soft_match"] if headers else "soft_match_accuracy"
            ): soft_match_accuracies.get_std(),
        },
    }
    accuracies_to_save = accuracies_to_save if accuracies_to_save else {}
    for task, metrics in accuracy_metrics.items():
        if not accuracies_to_save.get(task):
            accuracies_to_save[task] = {"task_id": task}
        accuracies_to_save[task].update(metrics)
    return accuracies_to_save


def format_task_accuracies(
    accuracies_to_save: dict[str, dict],
    task_ids: list[str | int],
    exact_match_accuracies: Accuracy,
    soft_match_accuracies: Accuracy,
    exact_match_std: Metric,
    soft_match_std: Metric,
    headers: list | dict = None,
) -> dict[str | int, dict]:
    """
    Format accuracies for the split to save them later,
    including the mean accuracy for all tasks

    :param accuracies_to_save: the accuracies to save
    :param task_ids: the task ids
    :param exact_match_accuracies: the exact-match accuracies
    :param soft_match_accuracies: the soft-match accuracies
    :param exact_match_std: the standard deviation for the exact-match accuracies (per task)
    :param soft_match_std: the standard deviation for the soft-match accuracies (per task)
    :param headers: the headers for the accuracies
    :return: the mean accuracies of all tasks
    """
    zipped_data = zip(
        task_ids,
        exact_match_accuracies.all,
        soft_match_accuracies.all,
        exact_match_std.all,
        soft_match_std.all,
    )

    for row, em_acc, sm_acc, em_std, sm_std in zipped_data:
        if str(row) not in accuracies_to_save.keys():
            accuracies_to_save[str(row)] = {"task_id": row}

        accuracies_to_save[str(row)].update(
            {
                headers["exact_match_acc"]: em_acc,
                headers["soft_match_acc"]: sm_acc,
                headers["exact_match_std"]: em_std,
                headers["soft_match_std"]: sm_std,
            }
        )

    accuracies_to_save = format_accuracy_metrics(
        exact_match_accuracies,
        soft_match_accuracies,
        exact_match_std,
        soft_match_std,
        headers,
        accuracies_to_save,
    )
    return accuracies_to_save


def _select_metric(metric_values: dict[str, float], keyword: str) -> list[float]:
    """
    Select accuracies from a dictionary of accuracies based on the keyword.

    :param metric_values: the accuracies
    :param keyword: the keyword to select the accuracies
    :return: a list of accuracies by the keyword
    """
    return [value for type_, value in metric_values.items() if keyword in type_]


def calculate_mean_accuracies(
    accuracies_to_save: dict,
    mean_headers: dict,
) -> dict[str, dict]:
    """
    Calculate mean accuracies for all tasks and update accuracies.

    :param accuracies_to_save: the split accuracies
    :param mean_headers: the headers for the mean accuracies
    :return: None
    """
    for task_id, accuracies in accuracies_to_save.items():
        exact_match_accuracies = Accuracy(
            "exact_match", _select_metric(accuracies, "exact_match")
        )
        soft_match_accuracies = Accuracy(
            "soft_match", _select_metric(accuracies, "soft_match")
        )

        accuracies_to_save[task_id].update(
            {mean_headers["exact_match"]: exact_match_accuracies.get_mean()}
        )
        if len(soft_match_accuracies) > 1:
            accuracies_to_save[task_id].update(
                {mean_headers["soft_match"]: soft_match_accuracies.get_std()}
            )
    return accuracies_to_save
