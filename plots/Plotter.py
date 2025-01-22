from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    This class plots the data.
    """

    def __init__(self, result_path: Path, color_map: str = None):
        """
        Initialize the plotter.
        """
        if color_map is None:
            self.cmap = plt.get_cmap("tab10")
        else:
            self.cmap = plt.get_cmap(color_map)

        self.result_path: Path = result_path

        self.plot_counter_task = 0
        self.plot_counter_prompt = 0

    def _save_plot(
        self,
        y_label: str,
        x_label: str,
        file_name: str,
        plot_name_add: str,
    ) -> None:
        """
        Save the plot to a file.

        :param y_label: label for the y-axis, i.e. the type of data
        :param x_label: label for the x-axis, i.e. the data for testing
        :param file_name: name of the file
        :param plot_name_add: addition to the plot name

        :return: None
        """
        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight")
        else:
            label = y_label.lower().replace(" ", "_")
            plt.savefig(
                self.result_path
                / f"{plot_name_add}{label}_per_{x_label.lower()}_no_{self.plot_counter_task}.png",
                bbox_inches="tight",
            )

        self.plot_counter_prompt += 1
        plt.close()

    @staticmethod
    def _plot_general_details(
        x_label: str,
        y_label: str,
        max_x_len: int,
        plot_name_add: str,
        compare: bool = False,
    ) -> None:
        """
        Plot the general details of the plot, e.g. labels, title, and legend.

        :param x_label: label for the x-axis
        :param y_label: label for the y-axis
        :param max_x_len: maximum length of the x-axis
        :param plot_name_add: addition to the plot name
        :param compare: whether the data is compared
        :return: None
        """
        plt.xticks(range(1, max_x_len + 1))
        plt.xlabel(x_label)

        plt.ylim(bottom=0, top=1)
        type_of_data = " ".join([part.capitalize() for part in y_label.split("_")])
        plt.ylabel(type_of_data)

        title = f"{type_of_data} per {x_label}"
        if compare:
            title += " and prompt"
        if plot_name_add:
            title += f" ({plot_name_add.strip('_')})"
        plt.title(title)

        plt.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True
        )

    def plot_acc_per_task(
        self,
        acc_per_task: list,
        x_label: str = "Task",
        y_label: str = "Accuracy",
        file_name=None,
        plot_name_add: str = "",
    ) -> None:
        """
        Plot the accuracy per task.

        :param acc_per_task: list of accuracies per task. We assume that the list is ordered ascending by task.
        :param x_label: label for the x-axis
        :param y_label: label for the y-axis
        :param file_name: name of the plot
        :param plot_name_add: addition to the plot name
        :return: None
        """
        plt.figure(figsize=(10, 5))
        colors = self.cmap(np.linspace(0, 1, len(acc_per_task)))
        plt.plot(range(1, len(acc_per_task) + 1), acc_per_task, color=colors[0])

        self._plot_general_details(
            x_label, y_label, len(acc_per_task), plot_name_add, compare=False
        )
        self._save_plot(y_label, x_label, file_name, plot_name_add)

    def plot_acc_per_task_and_prompt(
        self,
        acc_per_prompt_task: dict[str, list],
        x_label: str = "Task",
        y_label: str = "Accuracy",
        file_name=None,
        plot_name_add: str = "",
    ) -> None:
        """
        Plot the accuracy per task and prompt.

        :param acc_per_prompt_task: dict of accuracies. The keys are the prompts, the values a list of accuracies per
        task.
        :param x_label: label for the x-axis
        :param y_label: label for the y-axis
        :param file_name: name of the plot
        :param plot_name_add: addition to the plot name
        :return: None
        """
        plt.figure(figsize=(15, 5))
        colors = self.cmap(np.linspace(0, 1, len(acc_per_prompt_task)))

        max_x_len = 0
        for (prompt, acc), color in zip(acc_per_prompt_task.items(), colors):
            if len(acc) > max_x_len:
                max_x_len = len(acc)
            plt.plot(range(1, len(acc) + 1), acc, label=prompt, color=color)

        self._plot_general_details(
            x_label, y_label, len(acc_per_prompt_task), plot_name_add, compare=True
        )
        self._save_plot(y_label, x_label, file_name, plot_name_add)
