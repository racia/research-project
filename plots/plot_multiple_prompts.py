# Description: Plot the metrics for multiple prompts  by given paths to compare them.
# Furthermore, the accuracy files should be in the same format:
# - The first line should contain the average accuracy for the "zero task".
# - The following lines should contain the strict and soft-match metrics for each task.

from __future__ import annotations

from pathlib import Path

from data.DataLoader import DataLoader
from evaluation.Metrics import Accuracy
from plots.Plotter import Plotter
from settings.config import DataSplits
from utils import AccuracyType, create_disambiguators, get_paths


def plot_from_paths(
    paths: list[Path] | list[str],
    accuracy_types: list[AccuracyType],
    result_path: str,
    prompt_type: str,
    split: DataSplits = "valid",
) -> None:
    """
    Plot the metrics for multiple prompts to compare them.
    The accuracy files should be in the same format:
    - The first line should contain the average accuracy for the "zero task".
    - The following lines should contain the strict and soft-match metrics for each task.

    :param paths: list of paths to the accuracy files
    :param accuracy_types: list of accuracy types to plot (a plot for each type)
    :param result_path: path to save the plots
    :param split: split of the data to plot for
    :param prompt_type: type of prompt (e.g. "ICL")

    :return: None
    """
    Path(result_path).mkdir(parents=True, exist_ok=True)
    plotter = Plotter(results_path=Path(result_path))
    loader = DataLoader()

    if type(paths[0]) is str:
        paths = [Path(path) for path in paths]

    disambiguators = create_disambiguators(paths)
    unique_disambiguators = set(
        [disambiguator for disambiguator in disambiguators if disambiguator]
    )
    if len(unique_disambiguators) > 1:
        print("Disambiguators found:", *unique_disambiguators, sep="\n", end="\n\n")

    for accuracy_type in getattr(accuracy_types, "exact_match", "soft_match"):
        accuracies = {}
        for path, disambiguator in zip(paths, disambiguators):
            data, _ = loader.load_results(
                results_paths=[str(path)],
                headers=["task_id", "exact_match_accuracy", "soft_match_accuracy"],
            )
            prompt_name = Path(path).stem.replace("prompt_", f"{disambiguator}_", 1)
            accuracies[prompt_name] = Accuracy(accuracy_type, data[accuracy_type])

        plotter.plot_acc_per_task_and_prompt(
            acc_per_prompt_task=accuracies,
            y_label=accuracy_type,
            plot_name_add=[split, prompt_type],
        )

        print(
            f"Plotted {prompt_type} metrics for",
            accuracy_type,
            "and",
            split,
            end="\n\n",
        )


def plot_from_directory(
    directory: str,
    accuracy_types: list[AccuracyType],
    result_path: str,
    prompt_type: str,
    split: DataSplits = "valid",
) -> None:
    """
    Plot the metrics for multiple prompts to compare them.
    The accuracy files should be in the same format:
    - The first line should contain the average accuracy for the "zero task".
    - The following lines should contain the strict and soft-match metrics for each task.

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
    split: DataSplits,
    accuracy_types: list[AccuracyType],
    result_path: str,
    prompt_type: str,
) -> None:
    """
    Run the plotting for multiple prompts.
    The paths can be either directories or files. If directories are provided, the function will plot the metrics
    for each prompt in the directory. If files are provided, the function will plot the metrics for each file.
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
        split=DataSplits(valid=True, test=False, train=False),
        accuracy_types=[
            AccuracyType(exact_match="exact_match"),
            AccuracyType(exact_match="soft_match"),
        ],
        prompt_type="type",
        result_path="result/path",
    )
