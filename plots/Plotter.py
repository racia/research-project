from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation.Metrics import Accuracy, Metric
from inference.Prompt import Prompt
from interpretability.utils import InterpretabilityResult


class Plotter:
    """
    This class plots the data.
    """

    def __init__(self, results_path: Path, color_map: str = None):
        """
        Initialize the plotter.

        :param results_path: path to save the results
        :param color_map: color map for the plots
        """
        if color_map is None:
            self.cmap = plt.get_cmap("tab10")
        else:
            self.cmap = plt.get_cmap(color_map)

        self.results_path: Path = results_path

        self.plot_counter_task: int = 0
        self.plot_counter_prompt: int = 0

    def _save_plot(
        self,
        y_label: str,
        x_label: str,
        file_name: str,
        plot_name_add: list[str],
    ) -> None:
        """
        Save the plot to a file.

        :param y_label: label for the y-axis, i.e. the type of data
        :param x_label: label for the x-axis, i.e. the data for testing
        :param file_name: name of the file
        :param plot_name_add: addition to the plot name

        :return: None
        """
        if not plot_name_add:
            plt.savefig(self.results_path / file_name, bbox_inches="tight")
        else:
            label = y_label.lower().replace(" ", "_")
            file_name = f"{'_'.join(plot_name_add)}_{label}_per_{x_label.lower()}_no_{self.plot_counter_task}.png"
            plt.savefig(self.results_path / file_name, bbox_inches="tight")

        self.plot_counter_prompt += 1
        plt.close()

    @staticmethod
    def _plot_general_details(
        x_label: str,
        y_label: str,
        max_x_len: int,
        plot_name_add: list[str],
        number_of_prompts: int,
        metr_types: int = 1,
        step: int|float = 1,
        min_x_len: int = 0,
    ) -> None:
        """
        Plot the general details of the plot, e.g. labels, title, and legend.

        :param x_label: label for the x-axis
        :param y_label: label for the y-axis
        :param max_x_len: maximum length of the x-axis
        :param plot_name_add: addition to the plot name
        :param number_of_prompts: number of prompts to plot, if more than 6,
                                  the legend is placed outside the plot
        :param metr_types: number of metrics to plot, if more than 6,
                                  the legend is placed outside the plot
        :param min_x_len: minimum length of the x-axis
        :return: None
        """
        if step >= 1:
            plt.xticks(range(min_x_len, max_x_len + 1, step))
        else:  
            plt.xticks(np.arange(0, max_x_len + 0.1, step))

        plt.xlabel(x_label)

        if y_label.lower() in ["accurac", "correct"]:
            y_ticks = np.arange(0, 1.1, 0.1)
            plt.ylim(bottom=0, top=1.1)
            plt.ylim(bottom=0, top=1)
        elif "attention" in y_label.lower():
            y_ticks = np.arange(0, 0.5, 0.1)
            plt.ylim(bottom=0, top=0.5)
            plt.ylim(bottom=0, top=0.6)
        elif "reasoning" in y_label.lower():
            y_ticks = np.arange(0, 1.1, 0.1)
            plt.ylim(bottom=0, top=1.1)
            plt.ylim(bottom=0, top=1)
        else:
            y_ticks = np.arange(0, 1.1, 0.1)
            plt.ylim(bottom=0, top=1.1)
            plt.ylim(bottom=0, top=1)
            
        type_of_data = " ".join([part.capitalize() for part in y_label.split("_")])
        plt.ylabel(type_of_data)

        plt.grid(which="both", linewidth=0.5, axis="y", linestyle="--")

        title = f"{type_of_data} per {x_label}"
        if number_of_prompts > 1:
            title += " and prompt"
        elif metr_types > 1:
            title += " and metric"

        if plot_name_add:
            title += f" ({'; '.join(plot_name_add)})"

        plt.title(title)

        if (number_of_prompts > 6 or metr_types > 6):
            legend = plt.legend(
                loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True
            )
        else:
            legend = plt.legend(bbox_to_anchor=(1.1, 1.05))

        leg_lines = legend.get_lines()
        leg_texts = legend.get_texts()
        plt.setp(leg_lines, linewidth=3)
        plt.setp(leg_texts, fontsize="x-large")


    def correlation_map(
        self,
        data: dict[str, dict[str, tuple]],
        level: str,
        version: str,
        file_name: str = None,
        id: int = 1
    ) -> None:
        """
        Draw a heat map with the given data.
        :param data: 2D numpy array with the data to plot
        :param level: level of the data, e.g. "task", "sample", "part"
        :param version: version of the data, e.g. "before", "after"
        :param file_name: name of the file to save the plot
        :param id: int id of the level
        :return: None
        """
        plt.figure(figsize=(12, 8))
        data = pd.DataFrame(
            {k: {k2: v2[0] for k2, v2 in v.items()} for k, v in data.items()},
            index=data.keys(),
        )
        data.fillna(0) # To display 0 instead of empty block
        axis = sns.heatmap(data, annot=True)
        cbar = axis.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5)
        plt.title(f"Attention Map for {level} {id} ({version})", fontsize=10)
        plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15)
        Path.mkdir(self.results_path / version, exist_ok=True, parents=True)
        plt.savefig(self.results_path / version / file_name)
        plt.close()

    def draw_heat(
        self,
        interpretability_result: InterpretabilityResult,
        x_label: str,
        task_id: int,
        sample_id: int,
        part_id: int,
        version: str = "after",
        title: str = "",
    ) -> None:
        """
        Draw a heat map with the interpretability attention scores for the current task.
        (Partly taken from https://arxiv.org/abs/2402.18344)

        :param interpretability_result: interpretability result with the attention scores, x and y tokens
        :param x_label: label for the x-axis
        :param task_id: task id
        :param sample_id: sample id
        :param part_id: part id
        :param version: whether the plot is created after the setting was applied to the model output
        :param title: title of the plot
        :return: None
        """
        x = interpretability_result.x_tokens
        y = interpretability_result.y_tokens
        scores = interpretability_result.attn_scores

        plt.figure(figsize=(12, 8))
        # to get comparable heatmaps, the max value of all plots should be the same (as much as possible)
        max_score = max(np.max(scores[1:]), 0.25)
        axis = sns.heatmap(scores[1:], cmap="rocket_r", vmin=0, vmax=max_score)

        y = y[1:]
        x_ticks = [i + 0.5 for i in range(len(x))]
        y_ticks = [i + 0.5 for i in range(len(y))]

        plt.xlabel(x_label, fontdict={"size": 10})
        plt.ylabel("Model Output Tokens", fontdict={"size": 10})

        plt.xticks(ticks=x_ticks, labels=x, fontsize=5, rotation=60, ha="right")
        plt.yticks(ticks=y_ticks, labels=y, fontsize=5, rotation=0)

        cbar = axis.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5)

        if title:
            plt.title(title, fontsize=10)
            plt.subplots_adjust(top=0.92)

        plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15)

        plot_subdirectory = self.results_path / version / "interpretability"
        Path.mkdir(plot_subdirectory, exist_ok=True, parents=True)
        verbosity = "aggr" if "sentence" in x_label.lower() else "ver"
        plt.savefig(
            plot_subdirectory
            / f"attn_map-{task_id}-{sample_id}-{part_id}-{verbosity}.pdf"
        )

        plt.close()

    def plot_acc_per_task(
        self,
        acc_per_task: Accuracy,
        x_label: str = "Task",
        y_label: str = "Accuracy",
        file_name=None,
        plot_name_add: list[str] = None,
    ) -> None:
        """
        Plot the accuracy per task.

        :param acc_per_task: list of metrics per task. We assume that the list is ordered ascending by task.
        :param x_label: label for the x-axis
        :param y_label: label for the y-axis
        :param file_name: name of the plot
        :param plot_name_add: addition to the plot name
        :return: None
        """
        plt.figure(figsize=(10, 5))
        colors = self.cmap(np.linspace(0, 1, len(acc_per_task)))
        plt.plot(range(1, len(acc_per_task) + 1), acc_per_task.all, color=colors[0])

        self._plot_general_details(x_label, y_label, max_x_len=len(acc_per_task), plot_name_add=plot_name_add)
        self._save_plot(y_label, x_label, file_name, plot_name_add)

    def plot_acc_per_task_and_prompt(
        self,
        acc_per_prompt_task: dict[str | Prompt, Accuracy | Metric],
        x_label: str = "Task",
        y_label: str = "Accuracy",
        file_name=None,
        plot_name_add: list[str] = None,
    ) -> None:
        """
        Plot the accuracy per task and prompt.

        :param acc_per_prompt_task: dict of metrics. The keys are the prompts, the values a list of metrics per
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
            if len(acc.all) > max_x_len:
                max_x_len = len(acc.all)
            x_data, y_data = range(1, len(acc.all) + 1), acc.all

            if len(x_data) != len(y_data):
                raise ValueError(
                    f"x and y must have the same first dimension, but have shapes {len(x_data)} and {len(y_data)}"
                )

            if not y_data:
                raise ValueError("y_data is empty")

            plt.plot(
                x_data,
                y_data,
                label=prompt if isinstance(prompt, str) else prompt.name,
                color=color,
            )

        self._plot_general_details(
            x_label,
            y_label,
            max_x_len=max_x_len,
            plot_name_add=plot_name_add,
            number_of_prompts=number_of_prompts,
        )
        self._save_plot(y_label, x_label, file_name, plot_name_add)

    def plot_acc_with_std(
        self,
        acc_per_prompt_task: dict[str | Prompt, Accuracy | Metric],
        x_label: str = "Task",
        y_label: str = "Accuracy",
        file_name=None,
        plot_name_add: list[str] = None,
    ) -> None:
        plt.figure(figsize=(15, 5))
        number_of_prompts = 0
        max_x_len = 0

        means = [
            np.array(v.all)
            for k, v in acc_per_prompt_task.items()
            if "std" not in k.lower()
        ]
        stds = [
            np.array(v.all)
            for k, v in acc_per_prompt_task.items()
            if "std" in k.lower()
        ]
        labels = [key for key in acc_per_prompt_task.keys() if not "std" in key.lower()]
        colors = self.cmap(np.linspace(0, 1, len(labels)))

        for prompt, mean, std, color in zip(labels, means, stds, colors):
            number_of_prompts += 1
            if len(mean) > max_x_len:
                max_x_len = len(mean)

            x_data = np.arange(1, len(mean) + 1)

            plt.plot(
                x_data,
                mean,
                label=prompt if isinstance(prompt, str) else prompt.name,
                color=color,
            )
            # Add standard deviation shading
            plt.fill_between(
                x_data,
                mean - std,
                mean + std,
                color=color,
                alpha=0.25,
            )

        self._plot_general_details(
            x_label,
            y_label,
            max_x_len=max_x_len,
            plot_name_add=plot_name_add,
            number_of_prompts=number_of_prompts,
        )
        self._save_plot(y_label, x_label, file_name, plot_name_add)

    def plot_correlation(
        self,
        x_data: dict[str | Prompt, Accuracy | Metric],
        y_data: list[float],
        x_label: str = "X",
        y_label: str = "Y",
        file_name=None,
        plot_name_add: list[str] = None,
        path_add: str = None,
    ) -> None:
        """
        Plot the correlation between two variables.

        :param x_data: Either acc_per_prompt_task or sample_part_lengths
        :param y_data: data for the y-axis, e.g. attention scores
        :param x_label: label for the x-axis
        :param y_label: label for the y-axis
        :param file_name: name of the plot
        :param plot_name_add: addition to the plot name
        :param path_add: addition to the path where the plot is saved
        :return: None
        """

        plt.figure(figsize=(15, 5))
        colors = self.cmap(np.linspace(0, 1, len(x_data)))

        number_of_prompts = 0
        max_x_len = 1
        metr_types = 0
        
        for (metr_type, metr), color in zip(x_data.items(), colors):
            # number_of_prompts += 1
            metr_types+=1
            # This covers both cases: Metric (i.e. length of sentences) and Accuracy 
            if max(metr.all) > max_x_len:
                max_x_len = max(metr.all) # Case sample_part_lenghts: Set to max value
            min_x_len = min(metr.all) if min(metr.all) > 2 else 0
            if len(metr) != len(y_data):
                raise ValueError(
                    f"x and y must have the same first dimension, but have shapes {len(metr)} and {len(y_data)}"
                )

            if not y_data:
                raise ValueError("y_data is empty")
        
            plt.scatter(
                metr,
                y=[y.get_mean() for y in y_data] if isinstance(y_data[0], Metric) else y_data,
                label=metr_type if isinstance(metr_type, str) else metr_type.name,
                color=color,
            )

        self._plot_general_details(
            x_label,
            y_label,
            max_x_len,
            plot_name_add,
            number_of_prompts=number_of_prompts,
            metr_types=metr_types,
            step=0.1 if max_x_len==1 else 1,
            min_x_len=min_x_len,
        )
        if path_add:
            file_name = path_add / file_name
            Path(self.results_path / path_add).mkdir(parents=True, exist_ok=True)
        self._save_plot(y_label, x_label, file_name, plot_name_add)