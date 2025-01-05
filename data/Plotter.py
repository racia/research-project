import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    This class plots the data.
    """

    def __init__(self, result_path: str, color_map: str = None):
        """
        Initialize the plotter.
        """
        if color_map is None:
            self.cmap = plt.get_cmap("tab10")
        else:
            self.cmap = plt.get_cmap(color_map)

        self.result_path = result_path

        self.plot_counter_task = 0
        self.plot_counter_prompt = 0

    def plot_acc_per_task(
            self,
            acc_per_task: list,
            x_label: str = "Task",
            y_label: str = "Accuracy",
            plot_name=None,
    ):
        """
        Plot the accuracy per task.

        :param acc_per_task: list of accuracies per task. We assume that the list is ordered ascending by task.
        """
        plt.figure(figsize=(10, 5))

        # make the plots prettier
        colors = self.cmap(np.linspace(0, 1, len(acc_per_task)))

        plt.plot(range(1, len(acc_per_task) + 1), acc_per_task, color=colors[0])
        plt.xticks(range(1, len(acc_per_task) + 1))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(y_label + " per " + x_label)
        if plot_name is not None:
            plt.savefig(plot_name)
        else:
            plt.savefig(
                f"{self.result_path}/acc_per_task_no{self.plot_counter_task}.png"
            )

        self.plot_counter_task += 1

    def plot_acc_per_task_and_prompt(
            self,
            acc_per_task: dict,
            x_label: str = "Task",
            y_label: str = "Accuracy",
            plot_name=None,
    ) -> None:
        """
        Plot the accuracy per task and prompt.

        :param acc_per_task: dict of accuracies. The keys are the prompts, the values a list of accuracies per task.
        :param x_label: label for the x-axis
        :param y_label: label for the y-axis
        :return: None
        """
        plt.figure(figsize=(10, 5))

        # make the plots prettier
        colors = self.cmap(np.linspace(0, 1, len(acc_per_task)))

        max_x_len = 0

        for (prompt, acc), color in zip(acc_per_task.items(), colors):
            if len(acc) > max_x_len:
                max_x_len = len(acc)
            plt.plot(range(1, len(acc) + 1), acc, label=prompt, color=color)

        plt.xticks(range(1, max_x_len + 1))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(y_label + " per " + x_label + " and prompt")
        plt.legend()

        if plot_name is not None:
            plt.savefig(plot_name)
        else:
            plt.savefig(
                f"{self.result_path}/acc_per_task_and_prompt_no{self.plot_counter_prompt}.png"
            )

        self.plot_counter_prompt += 1
