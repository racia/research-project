# Description: Plot the accuracies for multiple prompts  by given paths to compare them.
# Furthermore, the accuracy files should be in the same format:
# - The first line should contain the average accuracy for the "zero task".
# - The following lines should contain the strict and soft-match accuracies for each task.

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from data.DataLoader import DataLoader
from evaluation.Accuracy import Accuracy
from plots.Plotter import Plotter
from settings.config import DataSplits


@dataclass
class AccuracyType:
    exact_match: str = "exact_match_accuracy"
    soft_match: str = "soft_match_accuracy"


def get_paths(directory: str, keyword: str) -> list[str]:
    """
    Get all paths that contain a keyword in the name from all the run subdirectories.

    :param directory: path to the directory containing the run subdirectories
    :param keyword: keyword to search for in the path names

    :return: list of paths containing the keyword
    """
    paths = []
    for item in Path(directory).iterdir():
        if item.is_dir():
            paths.extend(get_paths(item, keyword))
        elif keyword in item.name and item.name.endswith(".csv"):
            paths.append(str(item))
    return paths


def find_difference_in_paths(paths: list[Path]) -> list[str]:
    """
    Find the difference in the paths to disambiguate them.

    :param paths: list of paths to the files

    :return: difference in the paths
    """
    names = [path.parent.name for path in paths]
    if len(set(names)) != len(names):
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


def plot_from_paths(
    paths: list[str],
    accuracy_types: list[Union[AccuracyType.exact_match, AccuracyType.soft_match]],
    result_path: str,
    prompt_type: str,
    split: Union[
        DataSplits.train, DataSplits.valid, DataSplits.test
    ] = DataSplits.valid,
) -> None:
    """
    Plot the accuracies for multiple prompts to compare them.
    The accuracy files should be in the same format:
    - The first line should contain the average accuracy for the "zero task".
    - The following lines should contain the strict and soft-match accuracies for each task.

    :param paths: list of paths to the accuracy files
    :param accuracy_types: list of accuracy types to plot (a plot for each type)
    :param result_path: path to save the plots
    :param split: split of the data to plot for
    :param prompt_type: type of prompt (e.g. "ICL")

    :return: None
    """
    Path(result_path).mkdir(parents=True, exist_ok=True)
    plotter = Plotter(result_path=Path(result_path))
    loader = DataLoader()

    paths = [Path(path) for path in paths]
    disambiguators = create_disambiguators(paths)
    unique_disambiguators = set(
        [disambiguator for disambiguator in disambiguators if disambiguator]
    )
    if len(unique_disambiguators) > 1:
        print("Disambiguators found:", *unique_disambiguators, sep="\n", end="\n\n")

    for accuracy_type in accuracy_types:
        accuracies = {}
        for path, disambiguator in zip(paths, disambiguators):
            data = loader.load_result_data(
                result_file_path=str(path),
                headers=["task_id", "exact_match_accuracy", "soft_match_accuracy"],
            )
            prompt_name = Path(path).stem.replace("prompt_", f"{disambiguator}_", 1)
            accuracies[prompt_name] = Accuracy(accuracy_type, data[accuracy_type])

        plotter.plot_acc_per_task_and_prompt(
            acc_per_prompt_task=accuracies,
            y_label=accuracy_type,
            plot_name_add=f"{split}_{prompt_type}_",
        )

        print(
            f"Plotted {prompt_type} accuracies for",
            accuracy_type,
            "and",
            split,
            end="\n\n",
        )


def plot_from_directory(
    directory: str,
    accuracy_types: list[Union[AccuracyType.exact_match, AccuracyType.soft_match]],
    result_path: str,
    prompt_type: str,
    split: Union[
        DataSplits.train, DataSplits.valid, DataSplits.test
    ] = DataSplits.valid,
) -> None:
    """
    Plot the accuracies for multiple prompts to compare them.
    The accuracy files should be in the same format:
    - The first line should contain the average accuracy for the "zero task".
    - The following lines should contain the strict and soft-match accuracies for each task.

    :param directory: directory with results that contain the runs with the desired accuracy files
    :param accuracy_types: list of accuracy types to plot (a plot for each type)
    :param result_path: path to save the plots
    :param split: split of the data to plot for
    :param prompt_type: type of prompt (e.g. "ICL")

    :return: None
    """
    paths = get_paths(directory, keyword="metrics")
    print("Found the following paths:")
    print(*paths, end="\n\n", sep="\n")

    if not paths:
        raise ValueError("No files with metrics found.")

    plot_from_paths(paths, accuracy_types, result_path, prompt_type, split)


def run(
    paths: list[str],
    split: Union[DataSplits.train, DataSplits.valid, DataSplits.test],
    accuracy_types: list[Union[AccuracyType.exact_match, AccuracyType.soft_match]],
    result_path: str,
    prompt_type: str,
) -> None:
    """
    Run the plotting for multiple prompts.
    The paths can be either directories or files. If directories are provided, the function will plot the accuracies
    for each prompt in the directory. If files are provided, the function will plot the accuracies for each file.
    Paths should be either directories or files containing the accuracy files, *not mixed*.

    :param paths: list of paths to directories or files
    :param split: split of the data to plot for
    :param accuracy_types: list of accuracy types to plot (a plot for each type)
    :param result_path: path to save the plots
    :param prompt_type: type of prompt (e.g. "ICL")
    :return: None
    """
    if not paths:
        raise ValueError("No paths provided.")

    print("Running the plotting for multiple prompts.")

    if Path(paths[0]).is_dir():
        for path in paths:
            print(f"Plotting from directory: {path}")
            plot_from_directory(
                directory=path,
                split=split,
                accuracy_types=accuracy_types,
                prompt_type=prompt_type,
                result_path=result_path,
            )
    elif Path(paths[0]).is_file():
        plot_from_paths(
            paths=paths,
            split=split,
            accuracy_types=accuracy_types,
            prompt_type=prompt_type,
            result_path=result_path,
        )
    else:
        raise ValueError("Invalid paths provided.")

    print("Plots have been saved.")


if __name__ == "__main__":
    paths = [
        "path/to/accuracy/file/1",
        "or/path/to/directory",
    ]
    run(
        paths=paths,
        split=DataSplits.valid,
        accuracy_types=[AccuracyType.exact_match, AccuracyType.soft_match],
        prompt_type="type",
        result_path="result/path",
    )
