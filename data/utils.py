from __future__ import annotations

import os
from collections import defaultdict
from mailbox import FormatError
from pathlib import Path

from evaluation.Metrics import Accuracy, Metric
from inference.DataLevels import Features, SamplePart


def load_scenery(
    word_types: tuple[str, ...] = (
        "attr",
        "loc",
        "nh-subj",
        "obj",
        "part",
        "rel",
        "subj-attr",
        "subj",
        "other",
        "base_phrasal_verbs",
    ),
) -> set:
    """
    Get scenery words from the scenery_words folder and the Scenery base phrases verbs.
    Additionally, adds Scenery base phrasal words.

    :return: set of scenery words for filtering attention scores
    """
    scenery_words = set()
    for entry in os.scandir("data/scenery_words"):
        word_type = entry.name.strip(".txt")
        if word_type in word_types:
            with open(entry.path, "r", encoding="UTF-8") as f:
                scenery_words.update(f.read().splitlines())
    return scenery_words


def expand_cardinal_points(abbr_news: list[str]) -> list[str]:
    """
    Expands the abbreviations of cardinal points into full words by checking
    if any word as a list item belongs to possible abbreviations.

    :param abbr_news: list of possible abbreviations
    :return: list of words with cardinal points expanded with order preserved
    """
    cardinal_points = {"n": "north", "e": "east", "w": "west", "s": "south"}
    expanded_news = []
    for abbr in abbr_news:
        if abbr in cardinal_points.keys():
            expanded_news.append(cardinal_points[abbr])
        else:
            expanded_news.append(abbr)
    return expanded_news


def is_empty_file(file_path: Path) -> bool:
    """
    Checks if the file exists and is empty.

    :param file_path: the file path to check
    :return: True if file exists and is empty
             False if file is non-empty
    """
    return os.path.isfile(file_path) and os.path.getsize(file_path) == 0


def prepare_accuracy_headers(prompt_name: str = "", version: str = "after") -> dict:
    """
    Prepare the headers for the accuracies.

    :param prompt_name: the name of the prompt
    :param version: "after" if the metrics are calculated after the setting was applied, else "before"
    :return: the headers for the accuracies
    """
    prompt_name_ = prompt_name.replace("prompt_", "")
    return {
        "exact_match_acc": f"{prompt_name_}_exact_match_accuracy_{version}".strip("_"),
        "soft_match_acc": f"{prompt_name_}_soft_match_accuracy_{version}".strip("_"),
        "exact_match_std": f"{prompt_name_}_exact_match_std_{version}".strip("_"),
        "soft_match_std": f"{prompt_name_}_soft_match_std_{version}".strip("_"),
    }


def format_split_metrics(
    features: Features, headers: dict, metrics_to_save: dict, version: str = "after"
) -> dict[str, dict]:
    """
    Format the metrics for the split to save them later.

    :param features: the features of the split
    :param headers: accuracy headers
    :param metrics_to_save: the accuracies to save
    :param version: if the metrics are calculated after the setting was applied
    :return: the metrics to save
    """
    metrics = {
        f"there_{version}": features.there,
        f"verbs_{version}": features.verbs,
        f"pronouns_{version}": features.pronouns,
        f"not_mentioned_{version}": features.not_mentioned,
    }
    for metric, value in metrics.items():
        if metric not in metrics_to_save:
            metrics_to_save[metric] = {"task_id": metric}
        metrics_to_save[metric].update({headers["exact_match_acc"]: value})
    return metrics_to_save


def format_accuracy_metrics(
    exact_match_accuracies: Accuracy,
    soft_match_accuracies: Accuracy,
    exact_match_std: Metric,
    soft_match_std: Metric,
    headers: dict = None,
    accuracies_to_save: dict[str, dict] = None,
    version: str = "after",
) -> dict[str, dict]:
    """
    Format the accuracy metrics for the split to save them later:
    - mean accuracy for all tasks
    - standard deviation for all tasks

    :param exact_match_accuracies: the exact-match accuracies
    :param soft_match_accuracies: the soft-match accuracies
    :param exact_match_std: the standard deviation for the exact-match accuracies (per task)
    :param soft_match_std: the standard deviation for the soft-match accuracies (per task)
    :param headers: the headers for the data
    :param accuracies_to_save: the accuracies to save
    :param version: if the metrics are calculated after the setting was applied
    """
    accuracy_metrics = {
        "mean": {},
        "std": {},
    }
    exact_match_acc = (
        headers["exact_match_acc"] if headers else f"exact_match_accuracy_{version}"
    )
    soft_match_acc = (
        headers["soft_match_acc"] if headers else f"soft_match_accuracy_{version}"
    )
    em_std = headers["exact_match_std"] if headers else f"exact_match_std_{version}"
    sm_std = headers["soft_match_std"] if headers else f"soft_match_std_{version}"
    accuracy_metrics["mean"].update(
        {
            exact_match_acc: exact_match_accuracies.get_mean(),
            soft_match_acc: soft_match_accuracies.get_mean(),
            em_std: exact_match_std.get_mean(),
            sm_std: soft_match_std.get_mean(),
        }
    )
    accuracy_metrics["std"].update(
        {
            exact_match_acc: exact_match_accuracies.get_std(),
            soft_match_acc: soft_match_accuracies.get_std(),
        }
    )
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
    Select accuracies from a dictionary of accuracies based on the type_.

    :param metric_values: the accuracies
    :param keyword: the type_ to select the accuracies
    :return: a list of accuracies by the type_
    """
    return [value for type_, value in metric_values.items() if keyword in type_]


def calculate_mean_accuracies(
    accuracies_to_save: dict,
    mean_headers: dict,
    version: str = "after",
) -> dict[str, dict]:
    """
    Calculate mean accuracies for all tasks and update accuracies.

    :param accuracies_to_save: the split accuracies
    :param mean_headers: the headers for the mean accuracies
    :param version: if the metrics are calculated after the setting was applied
    :return: None
    """
    exact_match = f"exact_match_{version}"
    soft_match = f"soft_match_{version}"
    for task_id, accuracies in accuracies_to_save.items():
        exact_match_accuracies = Accuracy(
            "exact_match", _select_metric(accuracies, exact_match)
        )
        soft_match_accuracies = Accuracy(
            "soft_match", _select_metric(accuracies, soft_match)
        )

        accuracies_to_save[task_id].update(
            {mean_headers["exact_match_acc"]: exact_match_accuracies.get_mean()}
        )
        if len(soft_match_accuracies) > 1:
            accuracies_to_save[task_id].update(
                {mean_headers["soft_match_acc"]: soft_match_accuracies.get_std()}
            )
    return accuracies_to_save


def get_real_value(entry: str) -> str | int | float | bool | None:
    """
    Get the real value of the entry.

    :param entry: the entry to get the real value
    :return: the real value of the entry
    """
    if not entry:
        return None
    if entry.isdigit():
        return int(entry)
    elif entry.replace(".", "", 1).isdigit():
        return float(entry)
    elif entry == "None":
        return None
    elif entry == "True":
        return True
    elif entry == "False":
        return False
    else:
        return entry


def level_down(level_id: str) -> str | None:
    """
    Convert the level id to one level down. If it's already at the lowest level, return None.

    :param level_id: the level id to convert
    :return: the lower case level id
    """
    if "task" in level_id:
        return level_id.replace("task", "sample")
    elif "sample" in level_id:
        return level_id.replace("sample", "part")
    return None


def structure_parts(
    parts: list[SamplePart], level_id: str | None = "task_id"
) -> (
    dict[int, dict[int, list[SamplePart]]]
    | dict[int, list[SamplePart]]
    | list[SamplePart]
):
    """
    Structure the parts of the sample into samples and tasks.

    :param parts: the parts of the sample
    :param level_id: the level name for id to structure the parts by
                    "task_id" or "sample_id"
    :return: the structured parts
    """
    if type(parts) == dict:
        raise FormatError(
            "Parts should be a list, not a dict. They might be already structured."
        )

    if level_id not in ["task_id", "sample_id", "part_id"]:
        raise ValueError(
            f"Invalid id_ value: {level_id}. Expected 'task_id' or 'sample_id'."
        )

    if level_id is None:
        return parts

    if level_id == "part_id":
        return sorted(parts, key=lambda p: getattr(p, "part_id"))

    assert type(parts) == list

    id_parts = defaultdict(list)
    for part in parts:
        id_parts[getattr(part, level_id)].append(part)

    for key, parts_ in id_parts.items():
        assert type(parts_) == list
        id_parts[key] = structure_parts(parts_, level_down(level_id))

    return dict(sorted(id_parts.items(), key=lambda p: p[0]))


def get_samples_per_task(data: list[dict]) -> int:
    """
    Get the maximum number of samples per task.
    :param data: the data to get the samples from
    :return: the maximum number of samples per task
    """
    tasks_samples = {}
    for row in data:
        if row["task_id"] not in tasks_samples.keys():
            tasks_samples[row["task_id"]] = []
        if row["sample_id"] not in tasks_samples[row["task_id"]]:
            tasks_samples[row["task_id"]].append(row["sample_id"])
    return len(max(tasks_samples.values(), key=len))
