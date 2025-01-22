from dataclasses import dataclass
from pathlib import Path
from typing import Union

from baseline.config.baseline_config import DataSplits
from data.DataLoader import DataLoader
from plots.Plotter import Plotter


@dataclass
class Accuracy:
    strict: str = "accuracy"
    soft_match: str = "soft_match_accuracy"


def get_paths(directory: str, keyword: str) -> list[Path]:
    """
    Get all paths that contain a keyword in the name from the run subdirectories.

    :param directory: path to the directory containing the run subdirectories
    :param keyword: keyword to search for in the path names

    :return: list of paths containing the keyword
    """
    paths = []
    time_runs = [subdir for subdir in Path(directory).iterdir() if subdir.is_dir()]

    for time_run in time_runs:
        prompt_runs = [subdir for subdir in time_run.iterdir() if subdir.is_dir()]

        for prompt_run in prompt_runs:
            for path in prompt_run.glob(f"*{keyword}*"):
                paths.append(path)

    return paths


def plot_multiple_prompts(
    directory: str,
    accuracy_types: list[Union[Accuracy.strict, Accuracy.soft_match]],
    result_path: str,
    prompt_type: str,
    split: Union[
        DataSplits.train, DataSplits.valid, DataSplits.test
    ] = DataSplits.valid,
) -> None:
    """
    Plot the accuracies for multiple prompts to compare them.
    It is expected that the paths and prompt names are in the same order.
    Furthermore, the accuracy files should be in the same format:
    - The first line should contain the average accuracy for the "zero task".
    - The following lines should contain the strict and soft-match accuracies for each task.

    :param directory: list of paths to the accuracy files
    :param accuracy_types: list of accuracy types to plot (a plot for each type)
    :param result_path: path to save the plots
    :param split: split of the data to plot for
    :param prompt_type: type of prompt (e.g. "ICL")

    :return: None
    """
    paths = get_paths(directory, keyword="accuracies")
    print("Found the following paths:")
    print(*paths, end="\n\n", sep="\n")
    plotter = Plotter(result_path=Path(result_path))

    for accuracy_type in accuracy_types:
        accuracies = {}
        for path in paths:
            data = DataLoader.load_result_data(
                path,
                headers=["task_id", "accuracy", "soft_match_accuracy"],
            )
            prompt_accuracies = data[accuracy_type]
            # remove the average accuracy on the first position ("zero task")
            prompt_accuracies.pop(0)
            prompt_name = Path(path).stem.replace("prompt_", "")
            accuracies[prompt_name] = data[accuracy_type]

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


if __name__ == "__main__":
    # Model of usage
    directories = []
    types = []
    for dir, type_ in zip(directories, types):
        plot_multiple_prompts(
            directory=dir,
            split=DataSplits.valid,
            accuracy_types=[Accuracy.strict, Accuracy.soft_match],
            result_path=dir,
            prompt_type=type_,
        )
