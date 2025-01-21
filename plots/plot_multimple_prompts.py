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


def plot_multiple_prompts(
    paths: list[str],
    prompt_names: list[str],
    accuracy_types: list[Union[Accuracy.strict, Accuracy.soft_match]],
    result_path: str,
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

    :param paths: list of paths to the accuracy files
    :param prompt_names: list of corresponding prompt names
    :param accuracy_types: list of accuracy types to plot (a plot for each type)
    :param result_path: path to save the plots
    :param split: split of the data to plot for

    :return: None
    """
    if len(paths) != len(prompt_names):
        raise ValueError("The number of paths should be equal to the number of names.")

    plotter = Plotter(result_path=Path(result_path))

    for accuracy_type in accuracy_types:
        accuracies = {}
        for prompt_name, path in zip(prompt_names, paths):
            data = DataLoader.load_accuracy_data(path)
            prompt_accuracies = data[accuracy_type]
            # remove the average accuracy on the first position ("zero task")
            prompt_accuracies.pop(0)
            accuracies[prompt_name] = data[accuracy_type]

        plotter.plot_acc_per_task_and_prompt(
            acc_per_prompt_task=accuracies,
            y_label=accuracy_type,
            plot_name_add=f"{split}_{accuracy_type}_",
        )

        print("Plotted the accuracies for", accuracy_type, "and", split)


if __name__ == "__main__":
    # Model of usage
    paths = [
        "path_1",
        "path_2",
        "path_3",
    ]
    plot_multiple_prompts(
        paths=paths,
        prompt_names=["prompt_1", "prompt_2", "prompt_3"],
        split=DataSplits.train,
        accuracy_types=[Accuracy.strict, Accuracy.soft_match],
        result_path="results",
    )
