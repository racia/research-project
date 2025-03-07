import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from evaluation.Accuracy import Accuracy


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
        number_of_prompts: int = 1,
    ) -> None:
        """
        Plot the general details of the plot, e.g. labels, title, and legend.

        :param x_label: label for the x-axis
        :param y_label: label for the y-axis
        :param max_x_len: maximum length of the x-axis
        :param plot_name_add: addition to the plot name
        :param number_of_prompts: number of prompts to plot, if more than 6,
                                  the legend is placed outside the plot
        :return: None
        """
        plt.xticks(range(1, max_x_len + 1))
        plt.xlabel(x_label)

        y_ticks = np.arange(0, 1.1, 0.1)
        plt.yticks(y_ticks)
        plt.ylim(bottom=0, top=1.1)

        plt.ylim(bottom=0, top=1)
        type_of_data = " ".join([part.capitalize() for part in y_label.split("_")])
        plt.ylabel(type_of_data)

        plt.grid(which="both", linewidth=0.5, axis="y", linestyle="--")

        title = f"{type_of_data} per {x_label}"
        if number_of_prompts > 1:
            title += " and prompt"

        if plot_name_add:
            title += f" ({plot_name_add.strip('_')})"

        plt.title(title)

        if number_of_prompts > 6:
            plt.legend(
                loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True
            )
        else:
            plt.legend(bbox_to_anchor=(1.1, 1.05))

    def draw_heat(self, index, x, y, scores, task_id, part_id):
        """
        Draw a heat map with the interpretability attention scores
        
        :param x: x values
        :param y: y values
        :param index: the index, here: x values
        :param result_path: path to save the results to
        :return: None
        """

        plt.figure(figsize=(12, 6)) 
        ax=sns.heatmap(scores, cmap="RdBu_r", center=0)
        x_ticks, y_ticks = list(range(len(x))), list(range(len(y)))
        x_ticks, y_ticks = [i+0.5 for i in x_ticks], [i+0.5 for i in y_ticks]
        plt.ylabel('Answer tokens', fontdict={'size':10})
        plt.xlabel('Question tokens', fontdict={'size':10})
        plt.yticks(ticks=y_ticks, labels=y, fontsize=5, rotation=0) # Cot
        plt.xticks(ticks=x_ticks, labels=index, fontsize=5, rotation=30, ha="right") # Options
        plt.subplots_adjust(left=0.15, right=0.99, top=0.98, bottom=0.15)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5)
        cbarlabels = cbar.ax.get_yticklabels() 
        self.result_path.mkdir(exist_ok=True)
        plt.savefig(os.path.join(str(self.result_path), f"task-{task_id}-{part_id}.pdf"))
        plt.close()
        

    def plot_acc_per_task(
        self,
        acc_per_task: Accuracy,
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
        plt.plot(range(1, len(acc_per_task) + 1), acc_per_task.all, color=colors[0])

        self._plot_general_details(x_label, y_label, len(acc_per_task), plot_name_add)
        self._save_plot(y_label, x_label, file_name, plot_name_add)

    def plot_acc_per_task_and_prompt(
        self,
        acc_per_prompt_task: dict[str, Accuracy],
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

        number_of_prompts = 0
        max_x_len = 0
        for (prompt, acc), color in zip(acc_per_prompt_task.items(), colors):
            number_of_prompts += 1
            if len(acc) > max_x_len:
                max_x_len = len(acc)

            x_data = range(1, len(acc.all) + 1)
            y_data = acc.all
            if len(x_data) != len(y_data):
                raise ValueError(
                    f"x and y must have the same first dimension, but have shapes {len(x_data)} and {len(y_data)}"
                )

            plt.plot(x_data, y_data, label=prompt, color=color)

        self._plot_general_details(
            x_label,
            y_label,
            max_x_len,
            plot_name_add,
            number_of_prompts=number_of_prompts,
        )
        self._save_plot(y_label, x_label, file_name, plot_name_add)

    def plot_accuracies(
        self,
        accuracies,
        soft_match_accuracies,
        additional_info="",
        compare_prompts=False,
    ):
        if compare_prompts:
            # Save accuracies of all prompts
            self.plot_acc_per_task_and_prompt(
                acc_per_prompt_task=accuracies,
                y_label="Accuracy",
                plot_name_add=additional_info,
            )
            self.plot_acc_per_task_and_prompt(
                acc_per_prompt_task=soft_match_accuracies,
                y_label="Soft Match Accuracy",
                plot_name_add=additional_info,
            )
        else:
            # Save accuracies of one prompts
            self.plot_acc_per_task(
                acc_per_task=accuracies,
                y_label="Accuracy",
                plot_name_add=additional_info,
            )
            self.plot_acc_per_task(
                acc_per_task=soft_match_accuracies,
                y_label="Soft Match Accuracy",
                plot_name_add=additional_info,
            )
