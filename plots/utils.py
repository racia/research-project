from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class Identifiers:
    """
    Class to hold identifiers for tasks, samples, and parts.
    It is used to group the identifiers by task, sample or part.
    """

    def __init__(self, values: list[tuple[int, ...]], case: str, setting: str = None):
        self.values = values
        if "reas" not in case and "ans" not in case:
            # if case not in ERROR_CASES:
            raise ValueError(f"Case '{case}' is not a valid error case.")
        self.case = case
        self.setting = setting

    def group_by(
        self, task: bool = None, sample: bool = None, part: bool = None
    ) -> dict[int, list[tuple]]:
        """
        Group the identifiers by task, sample or part.
        :param task: if True, group by task_id
        :param sample: if True, group by sample_id
        :param part: if True, group by part_id
        :return: dictionary with grouped identifiers by the level specified
        """
        targets = [task, sample, part]
        if not any(targets):
            raise ValueError("Must provide either task, sample or part")
        if targets.count(True) > 1:
            raise ValueError("Must provide only one of task, sample or part")

        index = targets.index(True)
        grouped_indices = defaultdict(list)
        for identifier in self.values:
            key = identifier[index]
            grouped_indices[key].append(identifier)
        return grouped_indices


@dataclass
class AccuracyType:
    exact_match: str = "exact_match_accuracy"
    soft_match: str = "soft_match_accuracy"


def get_paths(directory: str | Path, keyword: str, file_format="csv") -> list[Path]:
    """
    Get all paths that contain a type_ in the name from all the run subdirectories.

    :param directory: path to the directory containing the run subdirectories
    :param keyword: type_ to search for in the path names
    :param file_format: format of the files to search for

    :return: list of paths containing the type_
    """
    paths = []
    if type(directory) is str:
        directory = Path(directory)
    if directory.is_file():
        return [directory]
    for item in directory.iterdir():
        if item.is_dir():
            paths.extend(get_paths(item, keyword, file_format))
        if keyword in item.name and item.name.endswith(f"{file_format}"):
            paths.append(item)
    return paths


def find_difference_in_paths(paths: list[Path]) -> list[str]:
    """
    Find the difference in the paths to disambiguate them.

    :param paths: list of paths to the files

    :return: difference in the paths
    """
    paths = [Path(path) if isinstance(path, str) else path for path in paths]
    names = [path.stem for path in paths]
    names_set = set(names)
    if "" in names_set:
        names_set.remove("")
    if names_set and len(set(names)) < len(names):
        return find_difference_in_paths([path.parent for path in paths])
    return names


def create_disambiguators(paths: list[Path]) -> list[str]:
    """
    Create disambiguators for the paths.

    :param paths: list of paths to the files

    :return: list of disambiguators
    """
    differences = find_difference_in_paths(paths)
    disambiguators = []
    names = [path.name for path in paths]
    for path, difference in zip(paths, differences):
        if names.count(path.name) > 1:
            disambiguators.append(difference)
        else:
            disambiguators.append("")
    return disambiguators


def determine_colour_scheme(case: str) -> str:
    """
    Determine the colour scheme based on the error case.

    :param case: error case

    :return: colour scheme
    """
    correct = re.search(r"\bcorr", case.lower())
    incorrect = re.search(r"\bincorr", case.lower())
    if correct and incorrect:
        return "YlOrBr"
    elif correct:
        return "Greens"
    elif incorrect:
        return "Reds"
    else:
        raise ValueError(f"Case '{case}' is not a valid error case.")


def simple_case(case: str) -> bool:
    """
    Identify if the error case simple, i.e., accounts only for reasoning or answer errors separately.
    :param case: error case
    :return: True if the case is simple, False otherwise
    """
    case = case.lower()
    if "reas" in case and "ans" in case:
        return False
    if "reas" in case or "ans" in case:
        return True
    else:
        raise ValueError(f"Case '{case}' is not a valid error case.")


def prepare_for_display_pie(labels: list[str], sizes: list[int]) -> list[str]:
    """
    Prepare labels for display in pie charts.
    :param labels: list of labels for the pie chart
    :param sizes: list of sizes for the pie chart
    :return: list of formatted labels for display
    """
    display_labels = []
    # Format labels with line breaks if too many slices and hide labels for zero-size slices
    for i, (label, size) in enumerate(zip(labels, sizes)):
        if "," in label:
            display_labels.append(label.replace(", ", ",\n"))
        elif size > 0:
            display_labels.append(label)
        else:
            display_labels.append("")

    return display_labels


def match_cases(should_be_simple: bool, case: str) -> bool:
    """
    Check if the error case matches the expected complexity.

    :param should_be_simple: whether the case should be simple
    :param case: error case

    :return: True if the case matches the expected complexity, False otherwise
    """
    is_simple = simple_case(case)
    return is_simple == should_be_simple


def plot_task_map_grid(
    plt,
    ax,
    task: int,
    samples: list[int],
    parts: list[int],
    mask: np.array,
) -> None:
    """
    Plot a grid map for a specific task showing samples and parts with a mask overlay.

    :param plt: matplotlib pyplot module
    :param ax: matplotlib axis to plot on
    :param task: task identifier
    :param samples: list of sample identifiers
    :param parts: list of part identifiers
    :param mask: list of (sample, part) tuples indicating missing indices
    :return: None
    """
    # Gray overlay for missing indices
    ax.imshow(mask, cmap="binary", alpha=0.3, aspect="auto")
    ax.set_title(f"Task {task}")
    ax.set_xticks(np.arange(-0.5, len(parts) - 1, 1))
    ax.set_yticks(np.arange(-0.5, len(samples) - 1, 1))
    ax.grid(color="black", linewidth=0.4)
    ax.set_xticklabels([f"P{p}" for p in parts], fontsize=8)
    ax.set_yticklabels([f"S{s}" for s in samples], fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Set axis limits to crop unused space
    ax.set_xlim(-0.5, len(parts) - 0.5)
    ax.set_ylim(-0.5, len(samples) - 0.5)
