from __future__ import annotations

import itertools
import warnings
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Sized

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator, PercentFormatter

from evaluation.DistractorAttention import DistractorAttentionStats
from evaluation.Metrics import Accuracy, Metric
from evaluation.utils import CASES_2_LABELS, CASES_TO_SIMPLE_ANS, FLOAT_2_STR
from inference.DataLevels import Features
from inference.Prompt import Prompt
from interpretability.utils import InterpretabilityResult
from plots.utils import (
    Identifiers,
    determine_colour_scheme,
    extract_attention_by_correct,
    plot_task_map_grid,
    prepare_for_display_pie,
)


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

        self.case_color_map = {
            item: color
            for item, color in zip(
                CASES_2_LABELS.keys(),
                self.cmap(np.linspace(0, 1, len(CASES_2_LABELS))),
            )
        }
        self.case_color_map = {  # GPT version
            "ans_corr": "#FF6E19",  # pure orange
            "ans_incorr": "#F5CBA7",  # light orange
            "reas_corr": "#2874A6",  # pure blue
            "reas_incorr": "#AED6F1",  # light blue
            "ans_null": "#6E6E6E",
            "reas_null": "#6E6E6E",
            "ans_null_reas_null": "#D3D3D3",  # pure gray
            "ans_corr_reas_null": "#E67E22",  # grayish orange
            "ans_incorr_reas_null": "#F5CBA7",  # light grayish orange
            "ans_null_reas_corr": "#2874A6",  # grayish blue
            "ans_null_reas_incorr": "#AED6F1",  # grayish light blue
            "ans_corr_reas_corr": "#A56B2E",  # strong brown
            "ans_corr_reas_incorr": "#D49F7A",  # brownish orange
            "ans_incorr_reas_corr": "#6B8FA4",  # brownish blue
            "ans_incorr_reas_incorr": "#D7CEC3",  # light brown
        }
        self.case_color_map = {  # mathematical
            "ans_corr": "#FF6F1B",  # pure orange (255, 110, 25)
            "ans_incorr": "#FFAF6E",  # light orange (255, 175, 110)
            "reas_corr": "#196EFF",  # pure blue (25, 110, 255)
            "reas_incorr": "#6EAFFF",  # light blue (110, 175, 255)
            "ans_null": "#6E6E6E",
            "reas_null": "#6E6E6E",
            "ans_null_reas_null": "#6E6E6E",  # pure gray (110, 110, 110)
            "ans_corr_reas_null": "#8C6E64",  # grayish orange (140, 110, 100)
            "ans_incorr_reas_null": "#B49664",  # light grayish orange (180, 150, 100)
            "ans_null_reas_corr": "#646E8C",  # grayish blue (100, 110, 140)
            "ans_null_reas_incorr": "#6496B4",  # grayish light blue (100, 150, 180)
            "ans_corr_reas_corr": "#966E78",  # strong brown (150, 110, 120)
            "ans_corr_reas_incorr": "#B48C8C",  # brownish orange (180, 140, 140)
            "ans_incorr_reas_corr": "#918CB4",  # brownish blue (145, 140, 180)
            "ans_incorr_reas_incorr": "#BEAFAA",  # light brown (190, 175, 170)
        }
        self.case_color_map = {  # with green and red as a basis?
            "ans_corr": "Greens",
            "ans_incorr": "Reds",
            "ans_null": "Greys",
            "reas_corr": "#196EFF",
            "reas_incorr": "#6EAFFF",
            "reas_null": "#6E6E6E",
            "ans_null_reas_null": "#6E6E6E",  # pure gray (110, 110, 110)
            "ans_corr_reas_null": "#6E6E96",  # grayish purple (110, 110, 150)
            "ans_incorr_reas_null": "#B49664",  # grayish blue (110, 140, 175)
            "ans_null_reas_corr": "#6E7864",  # grayish green (110, 120, 100)
            "ans_null_reas_incorr": "#78646E",  # grayish purple (120, 100, 110)
            "ans_corr_reas_corr": "#4B964B",  # dark green (75, 150, 75)
            "ans_corr_reas_incorr": "#966EAF",  # purple (150, 110, 175)
            "ans_incorr_reas_corr": "#508CAF",  # blue (80, 140, 175)
            "ans_incorr_reas_incorr": "#964B4B",  # dark red (150, 75, 75)
        }
        self.results_path: Path = results_path

        self.plot_counter_task: int = 0
        self.plot_counter_prompt: int = 0

        self.warning_counter = 0

    def _save_plot(
        self,
        y_label: str = None,
        x_label: str = None,
        file_name: str = None,
        plot_name_add: list[str] = None,
    ) -> None:
        """
        Save the plot to a file.

        :param x_label: label for the x-axis, i.e. the data for testing
        :param y_label: label for the y-axis, i.e. the type of data
        :param file_name: name of the file without the path and extension
        :param plot_name_add: addition to the plot name

        :return: None
        """
        if file_name:
            plt.savefig(self.results_path / file_name, bbox_inches="tight")
        elif x_label and y_label and plot_name_add:
            label = y_label.lower().replace(" ", "_")
            plt.savefig(
                self.results_path
                / f"{'_'.join(plot_name_add)}_{label}_per_{x_label.lower()}_no_{self.plot_counter_task}.png",
                bbox_inches="tight",
            )
        else:
            raise ValueError(
                "Either 'file_name' should be provided or 'x_label', 'y_label', and 'plot_name_add'."
            )

        self.plot_counter_prompt += 1
        plt.close()

    def _plot_general_details(
        self,
        x_label: str,
        y_label: str,
        max_x_len: int,
        plot_name_add: list[str],
        num_of_data_arrays: int,
        displ_percentage: bool = False,
        metr_types: int = 1,
        step: int | float = None,
        min_x_len: int = 0,
        legend_title: str = None,
    ) -> None:
        """
        Plot the general details of the plot, e.g. labels, title, and legend.

        :param x_label: label for the x-axis
        :param y_label: label for the y-axis
        :param max_x_len: maximum length of the x-axis
        :param plot_name_add: addition to the plot name
        :param num_of_data_arrays: number of arrays to plot, if more than 6,
                                  the legend is placed outside the plot
        :param displ_percentage: whether to display the y-axis as percentage
        :param metr_types: number of metrics to plot, if more than 6,
                                  the legend is placed outside the plot
        :param min_x_len: minimum length of the x-axis
        :step: step size for x-ticks
        :min_x_len: minimum length of the x-axis
        :return: None
        """
        try:
            if step >= 1:
                plt.xticks(range(min_x_len, max_x_len + 1, step))
            elif step > 0:
                plt.xticks(np.arange(0, max_x_len + 0.1, step))
            elif step == 0:
                pass
            elif step < 0:  # negative step
                raise ValueError(f"Step size must be non-negative, got <step={step}>")
        except TypeError:
            if not self.warning_counter:
                warnings.warn(f"No step size provided, defaulting to automatic ticks.")
            self.warning_counter += 1

        plt.xlabel(x_label)

        y_ticks = np.arange(0, 1.1, 0.1)
        if "accurac" in y_label.lower():
            plt.ylim(bottom=0, top=1)
        elif "attention" in y_label.lower():
            y_ticks = np.arange(0, 0.09, 0.01)
            plt.ylim(bottom=0, top=0.1)
        elif "reasoning" in y_label.lower():
            plt.ylim(bottom=0, top=1)
        elif displ_percentage:
            plt.ylim(bottom=0, top=1.01)

        plt.yticks = y_ticks

        type_of_data = " ".join([part.capitalize() for part in y_label.split(" ")])
        plt.ylabel(type_of_data)

        plt.grid(which="both", linewidth=0.5, axis="y", linestyle="--")

        title = f"{type_of_data} per {x_label}"
        if num_of_data_arrays > 1:
            title += " and prompt"
        elif metr_types > 1:
            title += " and metric"

        if plot_name_add:
            title += f" ({'; '.join(plot_name_add)})"

        plt.title(title)
        if displ_percentage:
            plt.gca().yaxis.set_major_formatter(
                PercentFormatter(1)
            )  # 1 = scale of data (data range)
            plt.gca().yaxis.set_ticks(y_ticks)

        if num_of_data_arrays > 6 or metr_types > 6 or "attributes" in y_label.lower():
            plt.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fancybox=True,
                shadow=True,
                title=legend_title,
            )
        else:
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title=legend_title)

    def correlation_map(
        self,
        data: dict[str, dict[str, tuple]],
        level: str,
        version: str,
        file_name: str = None,
        id: int = 1,
        split_name: str = None,
        path_add: str = "",
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
        data.fillna(0)  # To display 0 instead of empty block
        axis = sns.heatmap(data, annot=True)
        cbar = axis.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5)
        # Display x labels diagonally
        plt.xticks(rotation=25, ha="right")

        plt.title(
            f"Correlation Map for {level} {split_name if split_name else id} ({version})",
            fontsize=12,
        )
        plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15)

        (self.results_path / path_add).mkdir(parents=True, exist_ok=True)
        plt.savefig(self.results_path / path_add / file_name)
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
        x_labels = interpretability_result.x_tokens
        y_labels = interpretability_result.y_tokens
        scores = interpretability_result.attn_scores

        plt.figure(figsize=(12, 8))
        # to get comparable heatmaps, the max value of all plots should be the same (as much as possible)
        max_score = max(np.max(scores[1:]), 0.25)
        ax = sns.heatmap(scores[1:], cmap="rocket_r", vmin=0, vmax=max_score)

        # x_labels = x
        # y_labels = y[1:]
        x_tick_values = [i + 0.5 for i in range(len(x_labels))]
        y_tick_values = [i + 0.5 for i in range(len(y_labels))]

        plt.xlabel(x_label, fontdict={"size": 10})
        plt.ylabel("Model Output Tokens", fontdict={"size": 10})

        # plt.xticks(ticks=x_ticks, labels=x, fontsize=5, rotation=60, ha="right")
        # plt.yticks(ticks=y_ticks, labels=y, fontsize=5, rotation=0)
        ax.set_xticks(x_tick_values)
        ax.set_xticklabels(x_labels, fontsize=5, rotation=60, ha="right")
        ax.set_yticks(y_tick_values[1:])
        ax.set_yticklabels(y_labels[1:], fontsize=5, rotation=0)

        cbar = ax.collections[0].colorbar
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
            / f"attn_map-{task_id}-{sample_id}-{part_id}-{verbosity}.png"
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

        self._plot_general_details(
            x_label,
            y_label,
            len(acc_per_task),
            plot_name_add,
            num_of_data_arrays=1,
            step=1,
        )
        self._save_plot(x_label, y_label, plot_name_add, file_name)

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

        num_of_data_arrays = 0
        max_x_len = 0
        for (prompt, acc), color in zip(acc_per_prompt_task.items(), colors):
            num_of_data_arrays += 1
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
            num_of_data_arrays=num_of_data_arrays,
            step=1,
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
        num_of_data_arrays = 0
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
            num_of_data_arrays += 1
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
            num_of_data_arrays=num_of_data_arrays,
            step=1,
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
        path_add: str = "",
        level: str = None,
        include_soft: bool = True,
        label_add: list[str] = [],
    ) -> None:
        """
        Plot the correlation between two variables.

        :param x_data: Either acc_per_prompt_task or seen_context_lengths
        :param y_data: data for the y-axis, e.g. attention scores
        :param x_label: label for the x-axis
        :param y_label: label for the y-axis
        :param file_name: name of the plot
        :param plot_name_add: addition to the plot name
        :param path_add: addition to the path where the plot is saved
        :param level: level of the data, e.g. "task", "sample", "part"
        :param include_soft: whether to include soft metrics in the plot
        :param label_add: addition to the data labels
        :return: None
        """
        if level == "split":
            plt.figure(figsize=(15, 5))
        else:
            plt.figure(figsize=(10, 5))

        num_of_data_arrays = 0
        max_x_len = 1
        metr_types = 0
        min_x_len = 0

        x_data_points = {
            k: v for k, v in x_data.items() if include_soft or "soft" not in k.lower()
        }
        x_err = [
            x_data_points.pop(k)
            for k in x_data
            if "std" in k.lower() and k in x_data_points
        ]
        colors = self.cmap(np.linspace(0, 1, len(x_data_points)), alpha=0.7)

        for (metr_type, metr), std_dev, color in zip_longest(
            x_data_points.items(), x_err, colors
        ):
            # number_of_prompts += 1
            num_of_data_arrays += 1
            metr_types += 1
            # This covers both cases: Metric (i.e. length of sentences) and Accuracy
            if max(metr.all) > max_x_len:
                max_x_len = max(metr.all)  # Case sample_part_lenghts: Set to max value
                step_size = 5 if max_x_len > 30 else 1
            min_x_len = min(metr.all) if min(metr.all) > 2 else min_x_len
            if len(metr) != len(y_data):
                raise ValueError(
                    f"x and y must have the same first dimension, but have shapes {len(metr)} and {len(y_data)}"
                )

            if not y_data:
                raise ValueError("y_data is empty")

            plt.errorbar(
                metr,
                y=(
                    [y.get_mean() for y in y_data]
                    if isinstance(y_data[0], Metric)
                    else y_data
                ),
                xerr=std_dev,
                fmt="o",
                capsize=4,
                label=(
                    "{}{}".format(
                        " ".join(metr_type.split("_")).title(),
                        "\nwith Std Dev" if x_err else "",
                    )
                    if isinstance(metr_type, str)
                    else metr_type.name
                ),
                color=color,
                zorder=3,
            )
            for i, label in enumerate(label_add):
                plt.annotate(label, (metr[i] + 0.001, y_data[i] + 0.001))

        self._plot_general_details(
            x_label,
            y_label,
            max_x_len,
            plot_name_add,
            num_of_data_arrays=num_of_data_arrays,
            metr_types=metr_types,
            step=0.1 if max_x_len == 1 else step_size,
            min_x_len=min_x_len,
        )
        if path_add:
            file_name = path_add / file_name.lower()
            Path(self.results_path / path_add).mkdir(parents=True, exist_ok=True)
        self._save_plot(y_label, x_label, file_name, plot_name_add)
        plt.close()

    def get_color_or_map(self, c: str):
        """
        Get the color or colormap for a given case.
        :param c: case string
        """
        color = self.case_color_map[c]
        if color.startswith("#"):
            return color
        else:
            cmap = cm.get_cmap(color)
            return cmap

    def plot_answer_type_per_part(
        self,
        error_cases_ids: dict[str, str],
        specification: dict[str, str],
        reasoning_scores: dict[tuple, float] = None,
    ) -> None:
        """
        Plot a map of answer types (and optionally reasoning scores) per sample
        and part of each task.

        - Default: color encodes combined answer+reasoning type.
        - If reasoning_scores provided: color encodes only answer, and reasoning score is
        written as text.
        """
        # === Setup ===
        use_reasoning_scores = reasoning_scores is not None
        if not reasoning_scores:
            warnings.warn(
                "No reasoning scores provided, plotting answer types without scores. "
                "To include reasoning scores, "
                "pass a dict of {(task, sample, part): score} to the 'reasoning_scores' argument."
            )
            use_reasoning_scores = False

        # Determine which answer categories to use
        min_score, max_score = 0.0, 1.0
        if use_reasoning_scores:
            answer_types = ["ans_corr", "ans_incorr", "ans_null"]
            max_score = max(reasoning_scores.values())
            min_score = min(reasoning_scores.values())
        else:
            # exclude simple answer/reasoning types
            answer_types = [
                key for key in self.case_color_map.keys() if key.count("_") > 1
            ]
        colors = [self.get_color_or_map(c) for c in answer_types]

        # Parse case IDs
        ids_cases = {}  # dict[tuple[int, int, int], str]
        for case, indices in error_cases_ids.items():
            for idx in indices:
                t, s, p = tuple(
                    map(int, idx.split("\t")[1:])
                )  # drop the strike-through id and convert to int
                if use_reasoning_scores:
                    ids_cases[(t, s, p)] = CASES_TO_SIMPLE_ANS[case]
                else:
                    ids_cases[(t, s, p)] = case

        tasks = sorted(set(i[0] for i in ids_cases.keys()))
        n_tasks = len(tasks)
        if n_tasks % 4 == 0:
            n_cols = min(4, n_tasks)
        elif n_tasks % 3 == 0:
            n_cols = min(3, n_tasks)
        elif n_tasks % 2 == 0:
            n_cols = min(2, n_tasks)
        else:
            n_cols = 1
        n_rows = int(np.ceil(n_tasks / n_cols))

        # === Figure setup ===
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), squeeze=False)

        for i, task in enumerate(tasks):
            ax = axes[i // n_cols][i % n_cols]

            # Collect unique samples and parts
            # resert order: samples descending, parts ascending
            samples = sorted({s for t, s, _ in ids_cases if t == task}, reverse=True)
            parts = sorted({p for t, _, p in ids_cases if t == task})

            if not samples:
                ax.set_visible(False)
                continue

            # Build either an integer heatmap or an RGBA image depending on mode
            rgba_img, heatmap = None, None
            if use_reasoning_scores:
                rgba_img = np.ones(
                    (len(samples), len(parts), 4), dtype=float
                )  # default white
                mask = np.zeros((len(samples), len(parts)), dtype=bool)
            else:
                heatmap = np.zeros((len(samples), len(parts)), dtype=int)
                mask = np.zeros_like(heatmap, dtype=bool)

            missing_scores = set()
            for s_idx, s in enumerate(samples):
                for p_idx, p in enumerate(parts):
                    idx = (task, s, p)
                    if idx not in ids_cases:
                        mask[s_idx, p_idx] = True
                        if use_reasoning_scores:
                            rgba_img[s_idx, p_idx] = (1, 1, 1, 1)
                    else:
                        case = ids_cases[idx]
                        if use_reasoning_scores and idx in reasoning_scores:
                            score = reasoning_scores[idx]
                            # Normalize score to [0, 1]
                            norm_score = (
                                (score - min_score) / (max_score - min_score)
                                if max_score > min_score
                                else 0
                            )
                            colormap = colors[answer_types.index(case)]
                            # Resolve colormap object
                            if isinstance(colormap, str) and colormap.startswith("#"):
                                rgba = mcolors.to_rgba(colormap)
                            else:
                                cmap_obj = (
                                    colormap
                                    if hasattr(colormap, "__call__")
                                    else cm.get_cmap(colormap)
                                )
                                # Avoid sampling the absolute minimal value (pure white) for colormaps
                                # that start from white (e.g. 'Greys'). Reserve pure white for absent values.
                                cmap_name = getattr(cmap_obj, "name", "").lower()
                                # min_sample = 0.15 if "grey" in cmap_name else 0.0
                                min_sample = 0.15
                                sample = min_sample + norm_score * (1.0 - min_sample)
                                rgba = cmap_obj(sample)
                            rgba_img[s_idx, p_idx] = rgba
                        if use_reasoning_scores and idx not in reasoning_scores:
                            missing_scores.add(idx)
                            rgba_img[s_idx, p_idx] = (0, 0, 0, 0)
                        elif not use_reasoning_scores:
                            # store integer index for categorical mapping
                            heatmap[s_idx, p_idx] = answer_types.index(case)

            if missing_scores:
                warnings.warn(
                    f"When plotting for {specification['score']}, reasoning score missing "
                    f"the following indices in reasoning_scores dict: "
                    f"{missing_scores}"
                )

            # Display appropriately
            if use_reasoning_scores:
                ax.imshow(rgba_img, aspect="auto")
            else:
                # build a list of displayable colors for ListedColormap
                cmap_colors = []
                for col in colors:
                    if isinstance(col, str) and col.startswith("#"):
                        cmap_colors.append(col)
                    elif hasattr(col, "__call__"):
                        cmap_colors.append(col(0.5))
                    else:
                        cmap_colors.append(col)
                ax.imshow(heatmap, cmap=ListedColormap(cmap_colors), aspect="auto")
            # Draw grid and labels
            plot_task_map_grid(plt, ax, task, samples, parts, mask)

            # Overlay reasoning scores if provided
            if use_reasoning_scores:
                for s_idx, s in enumerate(samples):
                    for p_idx, p in enumerate(parts):
                        idx = (task, s, p)
                        if idx in reasoning_scores and not mask[s_idx, p_idx]:
                            score = round(reasoning_scores[idx], 2)
                            ax.text(
                                p_idx,
                                s_idx,
                                f"{score:.2f}",
                                ha="center",
                                va="center",
                                color="black",
                                fontsize=8,
                                fontweight="medium",
                                zorder=5,
                            )

        # === Legend ===
        legend_labels = [CASES_2_LABELS[a].replace(", ", ",\n") for a in answer_types]
        # Resolve any colormap objects/names to a concrete RGBA color for legend markers
        legend_colors = []
        for col in colors:
            if isinstance(col, str):
                if col.startswith("#"):
                    legend_colors.append(col)
                else:
                    # treat as named color or colormap name
                    try:
                        legend_colors.append(mcolors.to_rgba(col))
                    except Exception:
                        legend_colors.append(cm.get_cmap(col)(0.5))
            elif callable(col):
                # colormap object or function-like; sample at midpoint
                legend_colors.append(col(0.5))
            else:
                # fallback: try to convert to RGBA
                try:
                    legend_colors.append(mcolors.to_rgba(col))
                except Exception:
                    legend_colors.append((0.5, 0.5, 0.5, 1.0))

        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=label,
                markerfacecolor=lc,
                markersize=10,
            )
            for label, lc in zip(legend_labels, legend_colors)
        ]
        fig.legend(
            handles, legend_labels, loc="center left", bbox_to_anchor=(1.02, 0.5)
        )
        fig.suptitle(
            f"Error Cases {' '.join(specification.values())}", fontsize=14, y=0.95
        )
        fig.tight_layout(rect=(0, 0, 0.9, 0.9))

        out_path = (
            self.results_path
            / specification.pop("version", "")
            / f"error_case_map_{'_'.join(specification.values())}.png"
        )
        fig.savefig(out_path, bbox_inches="tight")

    def plot_case_heatmap(
        self,
        ids_settings: dict[tuple, list[str]],
        case_type: str,
        all_indices: set[tuple] = None,
    ) -> None:
        """
        Plots a grid of subplots, one per task. Each subplot is a heatmap of samples x parts.
        Subplot size adapts to the max number of samples/parts for each task.
        Gray color for indices that are not present in all_indices.
        :param ids_settings: {identifier: [settings]}
        :param case_type: "incorrect" or "correct" (for color)
        :param all_indices: set of all possible (task, sample, part) tuples
        :return: None
        """
        ids = list(ids_settings.keys())
        if not ids:
            raise ValueError("No cases to plot, pass non-empty 'ids_settings'.")

        # Get all tasks
        tasks = (
            sorted(set(i[0] for i in all_indices))
            if all_indices
            else sorted(set(i[0] for i in ids))
        )
        n_tasks = len(tasks)
        n_cols = min(4, n_tasks)
        n_rows = int(np.ceil(n_tasks / n_cols))

        # Calculate max samples/parts per task
        task_samples = {task: set() for task in tasks}
        task_parts = {task: set() for task in tasks}
        indices = all_indices if all_indices else ids
        for t, s, p in indices:
            if t in task_samples:
                task_samples[t].add(s)
                task_parts[t].add(p)

        # Calculate subplot sizes
        square_size = 0.5
        subplot_widths = [len(task_parts[task]) * square_size for task in tasks]
        subplot_heights = [len(task_samples[task]) * square_size for task in tasks]

        # Calculate figure size
        fig_width = sum(subplot_widths[i] for i in range(n_cols))
        fig_height = sum(subplot_heights[i] for i in range(0, n_tasks, n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False
        )
        cmap = plt.cm.get_cmap(determine_colour_scheme(case_type), 5)
        cmap = cmap(np.arange(cmap.N))
        cmap[0] = np.array([1, 1, 1, 1])  # White for 0 settings
        cmap = ListedColormap(cmap)

        im_4, ax_4 = None, None
        for i, task in enumerate(tasks):
            ax = axes[i // n_cols][i % n_cols]
            # resert order: samples descending, parts ascending
            samples = sorted(task_samples[task], reverse=True)
            parts = sorted(task_parts[task])
            heatmap = np.zeros((len(samples), len(parts)), dtype=int)
            mask = np.zeros_like(heatmap, dtype=bool)
            for s_idx, sample in enumerate(samples):
                for p_idx, part in enumerate(parts):
                    idx = (task, sample, part)
                    if all_indices and idx not in all_indices:
                        mask[s_idx, p_idx] = True
                    else:
                        heatmap[s_idx, p_idx] = len(ids_settings.get(idx, []))
            im = ax.imshow(heatmap, cmap=cmap, aspect="equal", vmin=0, vmax=4)
            plot_task_map_grid(plt, ax, task, samples, parts, mask)
            if i == 3:
                im_4 = im
                ax_4 = ax

        cbar = fig.colorbar(im_4 or im, ax=ax_4 or ax, pad=0.04, cmap=cmap.name)
        cbar.set_label("Number of Settings", fontsize=8)
        cbar.set_ticks([0, 1, 2, 3, 4])

        fig.suptitle(
            f"[{CASES_2_LABELS[case_type]}] Number of Settings for Case",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self._save_plot(file_name=f"error_case_heatmap_{case_type}")

    def plot_error_histogram(
        self,
        cases: dict[str, list[tuple] | set[tuple]],
        group_by: str | bool,
        normalize: bool = False,
        setting: str = None,
    ) -> None:
        """
        Plots a histogram for the number of items in each group (task/sample/part),
        divided by error category using different colors.
        :param cases: dict of {error_case: indices}
        :param group_by: 'task', 'sample', or 'part'
        :param normalize: if True, normalize counts to percentages per group
        :param setting: the setting name for the title
        :return: None
        """
        if group_by not in ("task", "sample", "part", None):
            raise ValueError("group_by must be 'task', 'sample', or 'part'")
        error_categories = list(cases.keys())
        # Collect all group ids
        all_group_ids = set()
        group_counts = {cat: {} for cat in error_categories}
        for case, indices in cases.items():
            identifiers = Identifiers(list(indices), case)
            grouped_ids = identifiers.group_by(
                task=(group_by == "task"),
                sample=(group_by == "sample"),
                part=(group_by == "part"),
            )
            group_counts[case] = {
                group_id: len(identifiers)
                for group_id, identifiers in grouped_ids.items()
            }
            [all_group_ids.add(id_) for id_ in grouped_ids.keys()]
        all_group_ids = sorted(all_group_ids)
        # Prepare data for stacked bar plot
        data = []
        for case in error_categories:
            data.append([group_counts[case].get(gid, 0) for gid in all_group_ids])

        if normalize:
            # Normalize to percentages per group
            totals = [
                sum(data[j][i] for j in range(len(error_categories)))
                for i in range(len(all_group_ids))
            ]
            for j in range(len(error_categories)):
                data[j] = [
                    (data[j][i] / totals[i] * 100 if totals[i] > 0 else 0)
                    for i in range(len(all_group_ids))
                ]
        fig, ax = plt.subplots(figsize=(10, 6))
        bottom = [0] * len(all_group_ids)
        for i, case in enumerate(error_categories):
            ax.bar(
                all_group_ids,
                data[i],
                bottom=bottom,
                color=self.case_color_map[case],
                label=CASES_2_LABELS[case],
            )
            bottom = [b + d for b, d in zip(bottom, data[i])]
        ax.set_xticks(all_group_ids)
        ax.set_xlabel(f"{group_by.capitalize()} ID")
        ax.set_ylabel("Percentage of Items (%)" if normalize else "Number of Items")
        setting = setting.upper() if setting else "ALL SETTINGS"
        ax.set_title(
            f"Histogram of Items per {group_by.capitalize()} by Error Category [{setting}]"
        )
        ax.legend()
        plt.tight_layout()
        normalization = "normalized" if normalize else "absolute"
        self._save_plot(
            file_name=f"error_histogram_{normalization}_{setting.title().replace(' ', '_')}"
        )

    def plot_case_pie(
        self,
        cases_indices: dict,
        setting: str = None,
        unique: bool = False,
    ) -> None:
        """
        Plots a pie chart for always correct/incorrect answer/reasoning cases.
        :param cases_indices: dict with keys like 'always_corr_answer', 'always_incorr_answer', etc., values are counts
        :param setting: optional, name of the setting for the title
        :param unique: if True, indicates that the cases are always correct/incorrect ones
        :return: None
        """
        labels, sizes, colors_to_use = [], [], []
        for case, indices in cases_indices.items():
            labels.append(CASES_2_LABELS[case])
            sizes.append(len(indices) if isinstance(indices, Sized) else indices)
            colors_to_use.append(self.case_color_map[case])

        def autopct_func(pct):
            """Custom autopct to show percentage only if > 0"""
            return f"{pct:.1f}%" if pct > 0 else ""

        fig, ax = plt.subplots(figsize=(12, 6))
        wedges, _, _ = ax.pie(
            sizes,
            labels=prepare_for_display_pie(labels, sizes),
            autopct=autopct_func,
            startangle=90,
            counterclock=False,
            labeldistance=1.05,
            # rotatelabels=True,
            colors=colors_to_use,
        )
        # add the lines between the slices
        fractions = np.array(sizes) / np.sum(sizes)
        angles = np.cumsum(fractions) * 2 * np.pi
        for angle in angles:
            ax.plot(
                [0, np.sin(angle)],
                [0, np.cos(angle)],
                color="black",
                linestyle="-",
                linewidth=0.4,
            )

        cases_str = "Always Correct/Incorrect Cases" if unique else "Cases"
        setting = setting.upper() if setting else "ALL SETTINGS"
        ax.set_title(f"Proportion of {cases_str} [{setting}]")
        ax.legend(
            wedges,
            labels,
            loc="center left",
            bbox_to_anchor=(1.2, 0.55),
        )
        # Shrink plot area to make space for legend
        fig.subplots_adjust(right=0.9 if len(sizes) < 5 else 0.8)
        plt.tight_layout(rect=(0, 0, 0.8, 1))
        uniqueness = "_unique" if unique else "_all"
        setting = setting.title().replace(" ", "_")
        self._save_plot(file_name=f"error_case_pie{uniqueness}_{setting}")

    def plot_corr_hist(
        self,
        x_data: dict[str | Prompt, Accuracy | Metric],
        y_data: dict[str : list[float] | np.array] = None,
        x_label: str = "X",
        y_label: str = "Y",
        displ_percentage: bool = False,
        file_name: str = None,
        plot_name_add: list[str] = None,
        level: str = None,
        id: int = 1,
        path_add: str = None,
    ) -> None:
        """
        Plot the correlation between two variables as histogram, i.e. parts attributes per part lengths.
        Categories are obtained from x_data unique values, e.g. part lengths 1,2,3,4,...
        Values for each category are obtained from y_data values, e.g. parts_answer_correct [1,0,1,1,...],
        which are finally summed/averaged to display per label.
        :param x_data: The x data to plot as bar categories, i.e. seen_context_lengths
        :param y_data: The y_data of labels, corresponding to categories from x_data, i.e. parts_answer_correct
        :param x_label: The label for x-axis
        :param y_label: The label for y-axis
        :param displ_percentage: whether to display the y-axis as percentage
        :param file_name: name of the file
        :param plot_name_add: addition to the plot name
        :param path_add: addition to the path where the plot is saved
        :param level: level of the data, e.g. "task", "sample", "part"
        :param id: int id of the level
        :return: None
        """
        color_map = {
            v: color
            for v, color in zip(
                FLOAT_2_STR.values(),
                self.cmap(np.linspace(0, 0.2, len(FLOAT_2_STR)))[::-1],
            )
        }  # Colors according to length of label data

        df_data = {}
        for k, v in y_data.items():
            if isinstance(v, dict):
                df_data.update(v)
            df_data[k] = v

        if level == "split":  # bigger plots for splits
            fig, ax = plt.subplots(figsize=(15, 8))
            width = 0.6
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            width = 0.35

        df = pd.DataFrame(
            list(zip(*x_data.values(), *df_data.values())),
            columns=[x_label] + list(df_data.keys()),
        )
        label_column = (
            " ".join(df.columns[2].split("_")).title() if len(df.columns) > 2 else None
        )

        if "correct" in y_label.lower():  # e.g. parts_answer_correct
            if "answer_in_self" in df.columns[2]:
                df["parts_answer_in_self"] = df["parts_answer_in_self"].apply(
                    lambda x: FLOAT_2_STR[x].capitalize()
                )
            # Store sum of answers correct per seen context length
            parts_per_class = df.groupby([df.columns[0]], group_keys=True)[
                df.columns[1]
            ].transform("count")
            # Add column for ratio of correct answers per category and label
            correct_per_label = df.groupby(
                [df.columns[0], df.columns[2]], group_keys=True
            )[df.columns[1]].transform(lambda x: np.sum(x == 1))
            incorr_per_label = df.groupby(
                [df.columns[0], df.columns[2]], group_keys=True
            )[df.columns[1]].transform(lambda x: np.sum(x == 0))

            corr_ratio = f"{df.columns[1]}_Ratio"
            incorr_ratio = "Incorr_Ratio"
            df[corr_ratio] = correct_per_label / parts_per_class
            df[incorr_ratio] = incorr_per_label / parts_per_class
            label_column += " and Answer is [In]Correct"
        else:  # e.g. attn_on_target
            df[x_label] = df[x_label].round()

        for col_name in [
            f"parts_{feat}"
            for feat in ["attn_on_target", "max_supp_attn"]
            if f"parts_{feat}" in df.columns
        ]:
            df[col_name] = df[col_name].round(
                2
            )  # Ensure numeric values are rounded if needed
        max_x_len = max(df[x_label])
        step_size = 2 if max_x_len > 30 else 1

        pivot_ratios = df.pivot_table(
            values=[corr_ratio, incorr_ratio],
            sort=False,
            index=x_label,
            columns=df.columns[2],
            fill_value=0,
        )  # parts_answer_correct first
        pivot_ratios.sort_index(
            axis=1, level=1, inplace=True, sort_remaining=False
        )  # sort for labels
        bottom = np.zeros(len(pivot_ratios.index))

        for class_lab_col in pivot_ratios:
            ax.bar(
                pivot_ratios.index,
                pivot_ratios[class_lab_col],
                width=width,
                bottom=bottom,
                label=(
                    "[Incorrect] " + class_lab_col[1]
                    if "incorr" in class_lab_col[0].lower()
                    else "[Correct] " + class_lab_col[1]
                ),
                color=color_map[class_lab_col[1].lower()],
                alpha=0.4 if "incorr" in class_lab_col[0].lower() else None,
            )
            bottom += pivot_ratios[class_lab_col]
        self._plot_general_details(
            x_label=x_label,
            y_label=y_label,
            max_x_len=max_x_len,
            num_of_data_arrays=1,
            displ_percentage=displ_percentage,
            plot_name_add=plot_name_add,
            legend_title=label_column,
            step=step_size,
        )

        (self.results_path / path_add).mkdir(parents=True, exist_ok=True)
        self._save_plot(
            y_label=y_label,
            x_label=x_label,
            file_name=f"{path_add}/{file_name.lower()}",
        )
        plt.close()

    def plot_corr_boxplot(
        self,
        x_data: dict[str | Prompt, Accuracy | Metric],
        y_data: dict[str : list[float] | np.array] = None,
        x_label: str = "X",
        y_label: str = "Y",
        displ_percentage: bool = False,
        version: str = False,
        file_name: str = None,
        plot_name_add: list[str] = None,
        path_add: str = "",
        level: str = None,
    ) -> None:
        """
        Plot the correlation between two variables as boxplot, i.e. parts attributes per part lengths.
        Categories are obtained from x_data unique values, e.g. part lengths 1,2,3,4,...
        Values for each category are obtained from y_data values, e.g. parts_answer_correct [1,0,1,1,...],
        which are finally summed/averaged to display per label.
        :param x_data: The x data to plot as boxplot categories, i.e. seen_context_lengths
        :param y_data: The y_data of labels, corresponding to categories from x_data, i.e. parts_answer_correct
        :param x_label: The label for x-axis
        :param y_label: The label for y-axis
        :param displ_percentage: whether to display the y-axis as percentage
        :param version: version of the data, e.g. "before", "after"
        :param file_name: name of the file
        :param plot_name_add: addition to the plot name
        :param path_add: addition to the path where the plot is saved
        :param level: level of the data, e.g. "task", "sample", "part"
        :return: None
        """
        if level == "split":  # bigger plots for splits
            plt.figure(figsize=(12, 8))
        else:
            plt.figure(figsize=(10, 5))
        colors = self.cmap(np.linspace(0, 0.2, len(y_data[list(y_data)[-1]])))

        df_data = {}
        for y_keys, y_vals in y_data.items():
            if any(isinstance(y_vals, dict_type) for dict_type in [dict, defaultdict]):
                df_data.update(y_vals)
            else:
                df_data[y_keys] = y_vals

        df = pd.DataFrame(
            list(zip(*x_data.values(), *df_data.values())),
            columns=[x_label] + list(df_data.keys()),
        )

        def _feat_mapping(x: str) -> str:
            # Map feature indices to feature names

            mapping = dict(
                map(
                    lambda x: (x[0], x[1]),
                    zip(range(5), y_data["parts_features"].keys()),
                )
            )
            feat_str = [
                mapping.get(i, "False")
                for i, part in enumerate(x.split("-"))
                if part in ["True", "1"]
            ]
            feat_str = [f.rstrip(f"_{version}") for f in feat_str]
            return "-".join(feat_str) if feat_str else None

        # Combine parts features to single column
        label_order = None
        if "parts_features" in y_data:
            label_order = [
                " ".join('"-"'.join(comb).split("_")).title().join('""')
                for L in range(1, 3)
                for comb in itertools.combinations(Features.attrs, L)
            ]
            label_order.insert(0, "No Features")
            df["features_combined"] = ""
            for col in y_data["parts_features"].keys():
                df["features_combined"] += df[col].astype(str) + "-"
            df["Features present"] = df["features_combined"].apply(
                lambda x: _feat_mapping(x)
            )
            df["Features present"] = df["Features present"].fillna("No Features")
        elif "correct" in df.columns[2]:
            df["parts_answer_correct"] = df["parts_answer_correct"].map(
                {1: "True", 0: "False"}
            )
            label_order = ["True", "False"]
        label_column = (
            df.columns[-1] if "features_combined" in df.columns else df.columns[2]
        )
        df[f"{label_column}_"] = df[label_column].apply(
            lambda x: (
                " ".join(x.split("_")).capitalize().join('""')
                if x not in ["No Features", "True", "False"]
                else x
            )
        )
        df[x_label] = df[x_label].round()

        # this sns.boxplot returns an error:
        # UnboundLocalError: cannot access local variable 'boxprops' where it is not associated with a value
        # when the input is all-NaN/empty after filtering
        # ax = sns.boxplot(
        #     data=df,
        #     x=x_label,
        #     y=df.columns[1],
        #     hue=f"{label_column}_" if len(df.columns) > 2 else None,
        #     hue_order=label_order if len(df.columns) > 2 else None,
        # )
        hue_col = f"{label_column}_"
        use_hue = hue_col in df.columns and df[hue_col].nunique(dropna=True) > 1

        try:
            ax = sns.boxplot(
                data=df,
                x=x_label,
                y=df.columns[1],
                hue=hue_col if use_hue else None,
                hue_order=label_order if use_hue else None,
            )
        except UnboundLocalError:
            # If error occurs (e.g. due to all-NaN data), plot without hue as fallback
            warnings.warn(
                "Boxplot with hue failed (possibly due to all-NaN data), plotting without hue as fallback."
            )
            print("Data for boxplot:\n", df)
            ax = sns.boxplot(
                data=df,
                x=x_label,
                y=df.columns[1],
                hue=None,
            )
        # Add vertical lines separating x categories
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.xaxis.grid(True, which="minor", color="black", lw=1, ls=":")

        self._plot_general_details(
            x_label=x_label,
            y_label=y_label,
            max_x_len=len(x_data),
            num_of_data_arrays=1,
            displ_percentage=displ_percentage,
            plot_name_add=plot_name_add,
            legend_title=" ".join(label_column.split("_")).title(),
        )

        if path_add:
            Path(self.results_path / path_add).mkdir(parents=True, exist_ok=True)
        self._save_plot(
            y_label=y_label,
            x_label=x_label,
            file_name=f"{path_add}/{file_name.lower()}",
        )
        plt.close()

    def plot_distractor_attn_boxplot(
        self,
        stats: DistractorAttentionStats,
        version: str,
        plot_name_add: list[str] = None,
        path_add: Path = None,
    ) -> None:
        """
        Plot a four-box comparison of mean attention on distractor versus neutral
        context sentences, split by answer correctness.

        The two correctness conditions (correct, incorrect) are shown side by side,
        each with a box for distractor sentences and a box for neutral sentences.
        A larger gap between the distractor and neutral boxes for incorrect answers
        than for correct answers suggests the model is misled by distractors.

        :param stats: accumulated distractor attention records for this version
        :param version: "before" or "after", used in the output file name
        :param plot_name_add: additional strings appended to the plot title
        :param path_add: sub-path appended to the plotter's base results path
        """
        grouped = stats.as_grouped()

        role_color = {"distractor": "#d7604a", "neutral": "#4a90d7"}
        label_correct = {True: "Correct", False: "Incorrect"}
        ordering = [
            (True, "distractor"),
            (True, "neutral"),
            (False, "distractor"),
            (False, "neutral"),
        ]

        box_data, tick_labels, colors = [], [], []
        for correct, role in ordering:
            vals = grouped[correct].get(role, [])
            if vals:
                box_data.append(vals)
                tick_labels.append(f"{label_correct[correct]}\n({role})")
                colors.append(role_color[role])

        if not box_data:
            print(
                f"[plot_distractor_attn_boxplot] No data to plot for version='{version}'."
            )
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(box_data, patch_artist=True, notch=False, widths=0.5)

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        for element in ("whiskers", "caps", "fliers", "medians"):
            plt.setp(bp[element], color="#333333", linewidth=1.2)

        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_ylabel("Mean Attention on Sentence Role", fontsize=11)

        title = "Distractor vs Neutral Attention by Answer Correctness"
        if plot_name_add:
            title += f"  [{', '.join(plot_name_add)}]"
        ax.set_title(title, fontsize=11, pad=10)
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        fig.tight_layout()

        self._save_plot(
            file_name=f"distractor_attn_boxplot_{version}.png",
        )
        plt.close(fig)
        self.plot_counter_prompt += 1

    def plot_distractor_attn_per_task(
        self,
        stats: DistractorAttentionStats,
        version: str,
        plot_name_add: list[str] = None,
        path_add: Path = None,
    ) -> None:
        """
        Plot a grouped bar chart of mean distractor and neutral attention per task,
        with correct and incorrect conditions shown side by side.

        Task and correctness combinations that have no data are left blank rather
        than drawn as zero, to avoid misleading comparisons.

        :param stats: accumulated distractor attention records for this version
        :param version: "before" or "after", used in the output file name
        :param plot_name_add: additional strings appended to the plot title
        :param path_add: sub-path appended to the plotter's base results path
        """
        per_task = stats.as_per_task()
        task_ids = sorted(per_task.keys())
        if not task_ids:
            print(
                f"[plot_distractor_attn_per_task] No data to plot for version='{version}'."
            )
            return

        x = np.arange(len(task_ids))
        width = 0.18

        bar_spec = [
            (True, "distractor", -1.5, "#c0392b", "Correct / Distractor"),
            (True, "neutral", -0.5, "#2980b9", "Correct / Neutral"),
            (False, "distractor", 0.5, "#e67e22", "Incorrect / Distractor"),
            (False, "neutral", 1.5, "#7f8c8d", "Incorrect / Neutral"),
        ]

        fig, ax = plt.subplots(figsize=(max(8, len(task_ids) * 1.1), 5))

        for correct, role, offset_mult, color, label in bar_spec:
            means = [per_task[tid].get(correct, {}).get(role, None) for tid in task_ids]
            xs = [
                x[i] + offset_mult * width for i, m in enumerate(means) if m is not None
            ]
            heights = [m for m in means if m is not None]
            if xs:
                ax.bar(xs, heights, width, label=label, color=color, alpha=0.82)

        ax.set_xticks(x)
        ax.set_xticklabels([f"T{tid}" for tid in task_ids], fontsize=9)
        ax.set_ylabel("Mean Attention", fontsize=11)

        title = "Per-Task Distractor vs Neutral Attention"
        if plot_name_add:
            title += f"  [{', '.join(plot_name_add)}]"
        ax.set_title(title, fontsize=11, pad=10)
        ax.legend(fontsize=9, ncol=2, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

        self._save_plot(
            file_name=f"distractor_attn_per_task_{version}.png",
        )
        plt.close(fig)
        self.plot_counter_prompt += 1

    def plot_distractor_attn_scatter(
        self,
        stats: DistractorAttentionStats,
        version: str,
        plot_name_add: list[str] = None,
        path_add: Path = None,
    ) -> None:
        """
        Plot a scatter of mean attention on neutral context versus mean attention
        on distractor sentences per part, coloured by answer correctness.

        A diagonal reference line (y = x) is included; points above it indicate
        that the model allocated more attention to distractors than to neutral
        context for that part. Clustering of incorrect answers above the diagonal
        supports the hypothesis that distractor attention correlates with errors.

        Only parts that have both attn_distractor and attn_neutral are plotted.

        :param stats: accumulated distractor attention records for this version
        :param version: "before" or "after", used in the output file name
        :param plot_name_add: additional strings appended to the plot title
        :param path_add: sub-path appended to the plotter's base results path
        """
        scatter_data = stats.as_scatter_data()

        fig, ax = plt.subplots(figsize=(6, 6))

        plot_spec = [
            (True, "#2ecc71", "Correct", "o"),
            (False, "#e74c3c", "Incorrect", "^"),
        ]
        all_vals: list[float] = []

        for correct, color, label, marker in plot_spec:
            xvals = scatter_data[correct]["neutral"]
            yvals = scatter_data[correct]["distractor"]
            if xvals:
                ax.scatter(
                    xvals,
                    yvals,
                    alpha=0.55,
                    s=35,
                    color=color,
                    marker=marker,
                    label=f"{label} (n={len(xvals)})",
                    edgecolors="none",
                )
                all_vals.extend(xvals)
                all_vals.extend(yvals)

        if not all_vals:
            print(
                f"[plot_distractor_attn_scatter] No data to plot for version='{version}'."
            )
            plt.close(fig)
            return

        lo, hi = min(all_vals), max(all_vals)
        margin = (hi - lo) * 0.05
        ref = [lo - margin, hi + margin]
        ax.plot(ref, ref, "k--", linewidth=0.9, alpha=0.45, label="y = x")

        ax.set_xlabel("Mean Attention on Neutral Context", fontsize=11)
        ax.set_ylabel("Mean Attention on Distractor Sentences", fontsize=11)

        title = "Distractor vs Neutral Attention per Part"
        if plot_name_add:
            title += f"  [{', '.join(plot_name_add)}]"
        ax.set_title(title, fontsize=11, pad=10)
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", alpha=0.35)
        fig.tight_layout()

        self._save_plot(file_name=f"distractor_attn_scatter_{version}.png")
        plt.close(fig)
        self.plot_counter_prompt += 1

    def plot_supporting_attention(
        self,
        stats: DistractorAttentionStats,
        x_label: str = "answer_correct",
        y_label: str = "supporting_attention",
        plot_name_add: list[str] | None = None,
        version: str = "before",
    ):
        """
        Mean supporting attention for correct vs incorrect answers.
        """

        data = extract_attention_by_correct(stats)

        labels = ["Correct", "Incorrect"]

        supporting_means = [
            (
                np.mean([v for v in data[True]["supporting"] if v is not None])
                if any(v is not None for v in data[True]["supporting"])
                else np.nan
            ),
            (
                np.mean([v for v in data[False]["supporting"] if v is not None])
                if any(v is not None for v in data[False]["supporting"])
                else np.nan
            ),
        ]

        plt.figure()

        plt.bar(labels, supporting_means)
        plt.ylabel("Attention")
        plt.title("Supporting Attention")

        self._save_plot(
            x_label=x_label,
            y_label=y_label,
            file_name=f"supporting_attention_{version}.png",
            plot_name_add=plot_name_add,
        )

    def plot_distractor_supporting_ratio(
        self,
        stats: DistractorAttentionStats,
        x_label: str = "answer_correct",
        y_label: str = "distractor_supporting_ratio",
        plot_name_add: list[str] | None = None,
        eps: float = 1e-8,
        version: str = "before",
    ):
        """
        Distractor / Supporting attention ratio per sample.
        """

        ratios_correct = []
        ratios_incorrect = []

        for r in stats.records:
            if getattr(r, "attn_supporting", None) is None:
                continue

            if r.attn_supporting is None or r.attn_distractor is None:
                continue

            ratio = r.attn_distractor / (r.attn_supporting + eps)

            if r.answer_correct:
                ratios_correct.append(ratio)
            else:
                ratios_incorrect.append(ratio)

        plt.figure()

        plt.boxplot([ratios_correct, ratios_incorrect], labels=["Correct", "Incorrect"])
        plt.ylabel("Distractor / Supporting Attention")
        plt.title("Distractor-to-Supporting Attention Ratio")

        self._save_plot(
            x_label=x_label,
            y_label=y_label,
            plot_name_add=plot_name_add,
            file_name=f"distractor_supporting_ratio_{version}.png",
        )

    def plot_attention_triplet(
        self,
        stats: DistractorAttentionStats,
        x_label: str = "attention_role",
        y_label: str = "attention",
        plot_name_add: list[str] | None = None,
        version: str = "before",
    ):
        """
        Compare supporting vs distractor vs neutral attention.
        """

        data = extract_attention_by_correct(stats)

        categories = ["supporting", "distractor", "neutral"]
        x = np.arange(len(categories))
        width = 0.35

        means_correct = [
            (
                np.mean([v for v in data[True][c] if v is not None])
                if any(v is not None for v in data[True][c])
                else np.nan
            )
            for c in categories
        ]

        means_incorrect = [
            (
                np.mean([v for v in data[False][c] if v is not None])
                if any(v is not None for v in data[False][c])
                else np.nan
            )
            for c in categories
        ]

        plt.figure()

        plt.bar(x - width / 2, means_correct, width, label="Correct")
        plt.bar(x + width / 2, means_incorrect, width, label="Incorrect")

        plt.xticks(x, categories)
        plt.ylabel("Attention")
        plt.title("Attention Breakdown")
        plt.legend()

        self._save_plot(
            x_label=x_label,
            y_label=y_label,
            plot_name_add=plot_name_add,
            file_name=f"attention_triplet_{version}.png",
        )
