from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AccuracyType:
    exact_match: str = "exact_match_accuracy"
    soft_match: str = "soft_match_accuracy"


def get_paths(directory: str | Path, keyword: str, file_format="csv") -> list[Path]:
    """
    Get all paths that contain a keyword in the name from all the run subdirectories.

    :param directory: path to the directory containing the run subdirectories
    :param keyword: keyword to search for in the path names
    :param file_format: format of the files to search for

    :return: list of paths containing the keyword
    """
    paths = []
    if type(directory) is str:
        directory = Path(directory)
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
