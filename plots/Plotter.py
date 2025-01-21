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

    def plot_acc_per_task(
        self,
        acc_per_task: list,
        x_label: str = "Task",
        y_label: str = "Accuracy",
        plot_name=None,
        plot_name_add: str = "",
    ) -> None:
        """
        Plot the accuracy per task.

        :param acc_per_task: list of accuracies per task. We assume that the list is ordered ascending by task.
        :param x_label: label for the x-axis
        :param y_label: label for the y-axis
        :param plot_name: name of the plot
        :param plot_name_add: addition to the plot name
        :return: None
        """
        plt.figure(figsize=(10, 5))

        # make the plots prettier
        colors = self.cmap(np.linspace(0, 1, len(acc_per_task)))

        plt.plot(range(1, len(acc_per_task) + 1), acc_per_task, color=colors[0])
        plt.xticks(range(1, len(acc_per_task) + 1))
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.title(f"{y_label} per {x_label}")
        if plot_name_add:
            plt.title(f"{y_label} per {x_label} ({plot_name_add.strip('_')})")

        if plot_name is not None:
            plt.savefig(plot_name)
        else:
            label = y_label.lower().replace(" ", "_")
            plt.savefig(
                self.result_path
                / f"{plot_name_add}{label}_per_task_no_{self.plot_counter_task}.png"
            )

        self.plot_counter_task += 1
        plt.close()

    def plot_acc_per_task_and_prompt(
        self,
        acc_per_prompt_task: dict[str, list],
        x_label: str = "Task",
        y_label: str = "Accuracy",
        plot_name=None,
        plot_name_add: str = "",
    ) -> None:
        """
        Plot the accuracy per task and prompt.

        :param acc_per_prompt_task: dict of accuracies. The keys are the prompts, the values a list of accuracies per
        task.
        :param x_label: label for the x-axis
        :param y_label: label for the y-axis
        :param plot_name: name of the plot
        :param plot_name_add: addition to the plot name
        :return: None
        """
        plt.figure(figsize=(15, 5))

        # make the plots prettier
        colors = self.cmap(np.linspace(0, 1, len(acc_per_prompt_task)))

        max_x_len = 0

        for (prompt, acc), color in zip(acc_per_prompt_task.items(), colors):
            if len(acc) > max_x_len:
                max_x_len = len(acc)
            plt.plot(range(1, len(acc) + 1), acc, label=prompt, color=color)

        plt.xticks(range(1, max_x_len + 1))
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.title(
            f"{y_label} per {x_label} and prompt",
        )
        if plot_name_add:
            plt.title(f"{y_label} per {x_label} ({plot_name_add.strip('_')})")

        # Locate the legend on the right side in the free space
        plt.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fancybox=True,
            shadow=True,
        )

        plt.title(f"{y_label} per {x_label} and prompt")
        if plot_name_add:
            plt.title(f"{y_label} per {x_label} ({plot_name_add.strip('_')})")

        plt.legend()
        if plot_name is not None:
            plt.savefig(plot_name, bbox_inches="tight")
        else:
            label = y_label.lower().replace(" ", "_")
            plt.savefig(
                self.result_path
                / f"{plot_name_add}{label}_per_{x_label.lower()}_and_prompt_no_{self.plot_counter_task}.png",
                bbox_inches="tight",
            )

        self.plot_counter_prompt += 1
        plt.close()
