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
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter

from evaluation.Metrics import Accuracy, Metric
from evaluation.utils import CASES_2_LABELS, CASES_TO_SIMPLE_ANS, FLOAT_2_STR
from inference.DataLevels import Features
from inference.Prompt import Prompt
from interpretability.DistractorAttention import DistractorAttentionStats
from interpretability.utils import InterpretabilityResult
from plots.utils import (
    Identifiers,
    determine_colour_scheme,
    extract_attention_by_correct,
    plot_task_map_grid,
    prepare_for_display_pie,
    safe_mean,
)

RATIO_YMIN: float = 1e-2  # min log ratio shown (distractor / supporting)
RATIO_YMAX: float = 1e2  # max log ratio shown
# Padding factor applied on top of the data maximum for attention plots.
YPAD: float = 1.3


def attn_ylim(
    *value_collections,
    pad: float = YPAD,
    abs_min: float = 1e-6,
) -> tuple[float, float]:
    """
    Compute a tight (0, ceiling) y-limit for distractor-attention plots from
    the actual data values in *value_collections*.

    After row-normalisation, per-sentence attention is O(1/n_sentences), so
    a fixed ceiling of 1.0 wastes most of the plot area.  This helper collects
    all finite values, takes the maximum, applies a padding factor, and returns
    a (0, ceiling) tuple suitable for ax.set_ylim().

    Falls back to (0, 0.5) if no finite values are found.

    :param value_collections: any number of flat lists / arrays of floats
        (None entries and np.nan are ignored).
    :param pad: multiplicative headroom above the observed maximum (default 1.3).
    :param abs_min: minimum ceiling value so the axis is never degenerate.
    :return: (0.0, ceiling)
    """
    all_vals: list[float] = []
    for coll in value_collections:
        if coll is None:
            continue
        for v in coll if hasattr(coll, "__iter__") else [coll]:
            if v is not None and np.isfinite(float(v)):
                all_vals.append(float(v))
    all_vals = all_vals or [0.5]  # falling back to the default
    ceiling = max(abs_min, max(all_vals) * pad)
    return 0.0, ceiling


def margin_ylim(
    *value_collections,
    pad: float = YPAD,
    abs_min: float = 1e-6,
) -> tuple[float, float]:
    """
    Compute a symmetric (-ceiling, +ceiling) y-limit for distractor-margin
    plots (distractor − supporting attention) from the actual data values.

    Falls back to (-0.2, 0.2) if no finite values are found.

    :param value_collections: flat lists / arrays of margin floats
        (None entries and np.nan are ignored).
    :param pad: multiplicative headroom above the observed absolute maximum.
    :param abs_min: minimum ceiling so the axis is never degenerate.
    :return: (-ceiling, +ceiling)
    """
    all_vals: list[float] = []
    for coll in value_collections:
        if coll is None:
            continue
        for v in coll if hasattr(coll, "__iter__") else [coll]:
            if v is not None and np.isfinite(float(v)):
                all_vals.append(float(v))
    if not all_vals:
        return -0.2, 0.2
    ceiling = max(abs_min, max(abs(v) for v in all_vals) * pad)
    return -ceiling, ceiling


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
            # Sentence-role colours used by the distractor-attention plots:
            "supporting": "#2874A6",  # blue, matches reas_corr
            "distractor": "#E67E22",  # orange, matches ans_corr_reas_null
            "neutral": "#6E6E6E",  # gray, matches ans_null
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
            # Sentence-role colours used by the distractor-attention plots:
            "supporting": "#196EFF",  # blue, matches reas_corr
            "distractor": "#FF6F1B",  # orange, matches ans_corr
            "neutral": "#6E6E6E",  # gray, matches ans_null
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
            # Sentence-role colours used by the distractor-attention plots:
            "supporting": "#196EFF",  # blue, matches reas_corr
            "distractor": "Oranges",  # colormap, follows the family pattern of this variant
            "neutral": "#6E6E6E",  # gray, matches reas_null
        }
        self.results_path: Path = results_path

        self.plot_counter_prompt: int = 0

        self.warning_counter = 0

        # Resolve the five distractor-plot colours from self.case_color_map so
        # they match the rest of Plotter's colour scheme. Map values may be a
        # hex string (e.g. "#FF6E19") or a colormap name (e.g. "Greens"); we
        # normalise to a single hex either way via _resolve_case_color.
        self.color_correct: str = self._resolve_case_color("ans_corr")
        self.color_incorrect: str = self._resolve_case_color("ans_incorr")
        self.color_supporting: str = self._resolve_case_color("supporting")
        self.color_distractor: str = self._resolve_case_color("distractor")
        self.color_neutral: str = self._resolve_case_color("neutral")

    def _resolve_case_color(self, case: str, sample: float = 0.6) -> str:
        """
        Return a single hex colour for a given key in ``case_color_map``.

        Some entries are colormap names (e.g. ``"Greens"``) rather than hex
        strings; for those we sample the colormap at ``sample`` to get a
        representative middle-saturation colour. Hex values are returned
        unchanged. Missing keys fall back to a neutral gray.

        :param case: key into ``self.case_color_map``
        :param sample: position in the colormap to sample (only used for
                       colormap-named entries); 0.6 gives a strong but not
                       maximally saturated colour
        :return: a hex colour string
        """
        value = self.case_color_map.get(case)
        if value is None:
            return "#888888"
        if isinstance(value, str) and value.startswith("#"):
            return value
        try:
            cmap = cm.get_cmap(value)
            return mcolors.to_hex(cmap(sample))
        except (ValueError, TypeError):
            return "#888888"

    def _resolve_save_target(
        self,
        file_name: str,
        path_add: Path | str | None,
    ) -> str:
        """
        Combine ``path_add`` with ``file_name`` into a single relative path
        suitable for ``_save_plot``'s ``file_name`` argument, creating the
        target directory if necessary.

        ``_save_plot`` writes its file to ``self.results_path / file_name`` and
        does not understand a separate ``path_add``. To land plots inside a
        sub-folder we have to (a) make sure the sub-folder exists and (b)
        embed it in the ``file_name`` string. This helper centralises that
        routing so callers can keep using ``path_add`` as a clean argument.

        :param file_name: bare filename (no directory component)
        :param path_add: optional sub-path under ``self.results_path``;
                         ``None`` or empty means save directly under
                         ``self.results_path``
        :return: the combined ``"<path_add>/<file_name>"`` string, or just
                 ``file_name`` when no sub-path was given
        """
        if not path_add:
            return file_name
        target_dir = self.results_path / path_add
        target_dir.mkdir(parents=True, exist_ok=True)
        return f"{path_add}/{file_name}"

    def _save_plot(
        self,
        y_label: str = None,
        x_label: str = None,
        file_name: str = None,
        path_add: Path = None,
    ) -> None:
        """
        Save the plot to a file.

        :param x_label: label for the x-axis, i.e. the data for testing
        :param y_label: label for the y-axis, i.e. the type of data
        :param file_name: name of the file without the path and extension
        :param path_add: addition to the plot subdirectory, e.g. for different versions or levels

        :return: None
        """
        if file_name:
            file_name = str(file_name)
            if not Path(file_name).suffix:
                raise ValueError(
                    f"'file_name' must include a file-type suffix (e.g. '.png'), "
                    f"got: {file_name!r}"
                )
            plt.savefig(self.results_path / file_name, dpi=300, bbox_inches="tight")
        elif x_label and y_label and path_add:
            label = y_label.lower().replace(" ", "_")
            plt.savefig(
                self.results_path / path_add / f"{label}_per_{x_label.lower()}.png",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            raise ValueError(
                "Either 'file_name' should be provided or 'x_label', 'y_label', and 'path_add'."
            )

        self.plot_counter_prompt += 1
        plt.close()

    def _write_plot_data_txt(
        self,
        png_path: str,
        sections: list[tuple[str, list[tuple[str, float | int | None]]]],
    ) -> None:
        """
        Save the underlying numbers of a plot as a sibling ``.txt`` file.

        Given the relative PNG path that was passed to ``_save_plot`` (e.g.
        ``"distractor/distractor_attn_boxplot_before.png"``), this writes
        ``"distractor/distractor_attn_boxplot_before.txt"`` containing one
        ``key: value`` line per entry, grouped under section headers.

        Used by the distractor-attention plots so that exact medians, means,
        sample counts, etc. are available without printing them on top of the
        figure (which clutters the visual). The plots themselves accept a
        ``show_values`` flag to toggle in-plot annotations independently.

        :param png_path: relative path of the PNG (relative to results_path)
        :param sections: list of (section_title, [(label, value), ...]) tuples;
                         a value of None is rendered as "n/a"
        """
        txt_path = self.results_path / Path(png_path).with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for section_title, entries in sections:
            lines.append(f"# {section_title}")
            for label, value in entries:
                if value is None:
                    lines.append(f"{label}: n/a")
                elif isinstance(value, float):
                    lines.append(f"{label}: {value:.6f}")
                else:
                    lines.append(f"{label}: {value}")
            lines.append("")
        txt_path.write_text("\n".join(lines), encoding="utf-8")

    def _format_short_title(
        self,
        base: str,
        plot_name_add: list[str] | None,
    ) -> str:
        """
        Build a short, easy-to-read plot title.

        The original plots concatenated all ``plot_name_add`` entries verbatim
        into the title, which produced long titles like
        ``"... [my_prompt_v1, test, before]"``. This helper drops the version
        suffix (which is already encoded in the filename and the colour scheme)
        and keeps only the prompt/split tag in compact brackets.

        :param base: short headline of the plot (e.g. "Attention by role")
        :param plot_name_add: optional context tags from the caller
        :return: a one-line title suitable for ``ax.set_title``
        """
        if not plot_name_add:
            return base
        # Drop the version tag ("before"/"after") to avoid duplication: the
        # version is already in the filename and colour scheme.
        tags = [t for t in plot_name_add if t.lower() not in ("before", "after")]
        if not tags:
            return base
        return f"{base}  ({', '.join(tags)})"

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
        combined_handles: list[Line2D] = None,
        combined_labels: list[str] = None,
        legend_title: str = None,
        experiment: str = "direct_answer",
        num_parts: int = None,
        num_samples: int = None,
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
        :num_parts: The number of parts processed (if applicable, e.g. for part-level plots)
        :num_samples: The number of samples processed
        :return: None
        """
        try:
            if step >= 1:
                plt.xticks(range(min_x_len, max_x_len + 1, step))
            elif step > 0 and step < 1:
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

        if "diff" in y_label.lower():
            y_ticks = np.arange(-2.0, 2.1, 0.2)
            plt.ylim(bottom=-2.0, top=2.0)
        elif "accurac" in y_label.lower():
            plt.ylim(bottom=0, top=1)
        elif "attention" in y_label.lower():
            if "direct" in experiment.lower():
                # Attention scores vary between 0 and 1, with higher values in the direct answer setting. We dynamically adjust the y-axis limit to better visualize the differences between the scores, which are often below 0.6. For da scores, we keep the full range up to 1, as they can vary more widely.
                plt.ylim(bottom=0, top=1.0)
            else:
                y_ticks = np.arange(0, 0.31, 0.05)
                plt.ylim(bottom=0, top=0.3)
        elif "reasoning" in y_label.lower():
            plt.ylim(bottom=0, top=1)
        elif displ_percentage:
            plt.ylim(bottom=0, top=1.01)

        plt.yticks(y_ticks)

        type_of_data = " ".join(
            [
                part.capitalize() if part not in ["Per", "Of", "On"] else part
                for part in y_label.split(" ")
            ]
        )
        plt.ylabel(type_of_data)

        plt.grid(which="both", linewidth=0.5, axis="y", linestyle="--")
        plt.gca().set_axisbelow(True)

        title = f"{type_of_data} per {x_label}"
        if num_of_data_arrays > 1:
            title += " and prompt"
        elif metr_types > 1:
            title += " and metric"

        if plot_name_add:
            title += f" ({'; '.join(plot_name_add)})"

        plt.title(title)

        stats = []
        if num_parts is not None:
            stats.append(f"Processed parts: {num_parts}")
        if num_samples is not None:
            stats.append(f"Processed samples: {num_samples}")
        if stats:
            plt.gcf().text(
                0.99, 0.01, "\n".join(stats), fontsize=9, ha="right", va="bottom"
            )  # transform=plt.gca().transAxes for in-axes placement

        if displ_percentage:
            plt.gca().yaxis.set_major_formatter(
                PercentFormatter(1)
            )  # 1 = scale of data (data range)
            plt.gca().yaxis.set_ticks(y_ticks)

        if num_of_data_arrays > 6 or metr_types > 6 or "attributes" in y_label.lower():
            plt.legend(
                handles=combined_handles,
                labels=combined_labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fancybox=True,
                shadow=True,
                title=legend_title,
            )
        else:
            plt.legend(
                handles=combined_handles,
                labels=combined_labels,
                loc="upper left",
                bbox_to_anchor=(1, 1),
                title=legend_title,
            )

    def correlation_map(
        self,
        data: dict[str, dict[str, tuple]],
        level: str,
        version: str,
        file_name: str,
        id: int = 1,
        split_name: str = None,
        path_add: Path = None,
    ) -> None:
        """
        Draw a heat map with the given data.
        :param data: 2D numpy array with the data to plot
        :param level: level of the data, e.g. "task", "sample", "part"
        :param version: version of the data, e.g. "before", "after"
        :param file_name: name of the file to save the plot
        :param id: int id of the level
        :param split_name: name of the split, if level is "split"
        :param path_add: addition to the path where the plot is saved
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
        # Display x/ y labels diagonally
        axis.set_xticklabels(axis.get_xticklabels(), rotation=25, ha="right")
        axis.set_yticklabels(axis.get_yticklabels(), rotation=25, ha="right")

        plt.title(
            f"Correlation Map for {level} {split_name if split_name else id} ({version})",
            fontsize=12,
        )
        plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15)

        png_path = self._resolve_save_target(file_name, path_add)
        self._save_plot(file_name=png_path)

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
        y_labels = y_labels[1:]
        scores = interpretability_result.attn_scores
        scores = scores[1:]

        plt.figure(figsize=(12, 8))
        if len(scores) > 1:
            # to get comparable heatmaps, the max value of all plots should be the same (as much as possible)
            max_score = max(np.max(scores[1:]), 0.25)
        else:
            warnings.warn(
                f"No attention scores for task {task_id}, sample {sample_id}, part {part_id},"
                f" defaulting max_score to 0.25"
            )
            max_score = 0.25  # default
        ax = sns.heatmap(scores, cmap="rocket_r", vmin=0, vmax=max_score)

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
        ax.set_yticks(y_tick_values)
        ax.set_yticklabels(y_labels, fontsize=5, rotation=0)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5)

        if title:
            plt.title(title, fontsize=10)
            plt.subplots_adjust(top=0.92)

        plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15)

        plot_subdirectory = (
            self.results_path / version / f"Task-{task_id}" / "interpretability"
        )
        Path.mkdir(plot_subdirectory, exist_ok=True, parents=True)
        verbosity = "aggr" if "sentence" in x_label.lower() else "ver"
        plt.savefig(
            plot_subdirectory
            / f"attn_map-{task_id}-{sample_id}-{part_id}-{verbosity}.png",
            dpi=300,
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
        plt.plot(
            range(1, len(acc_per_task) + 1),
            acc_per_task.all,
            color=colors[0],
            alpha=0.82,
        )

        self._plot_general_details(
            x_label,
            y_label,
            len(acc_per_task),
            plot_name_add,
            num_of_data_arrays=1,
            step=1,
        )
        path_add = Path("/".join(plot_name_add)) if plot_name_add else None
        png_path = self._resolve_save_target(
            file_name
            or f"{y_label.lower().replace(' ', '_')}_per_{x_label.lower()}.png",
            path_add,
        )
        txt_rows = [(f"task={i}", float(v)) for i, v in enumerate(acc_per_task.all, 1)]
        self._write_plot_data_txt(png_path, [("Accuracy per task", txt_rows)])
        self._save_plot(file_name=png_path)

    def plot_acc_and_toxic_cot(
        self,
        group: str,
        df: pd.DataFrame,
        version: str,
        plot_name_add: list[str] | None = None,
    ):
        plt.close()
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # PLOT ACCURACY
        plot_df = (
            df.groupby(group)
            .agg(
                Direct_answer=(f"answer_correct_{version}_da", "mean"),
                With_reasoning=(f"answer_correct_{version}_reas", "mean"),
            )
            .reset_index()
            .melt(id_vars=group, var_name="condition", value_name="accuracy")
        )
        ax = sns.barplot(
            data=plot_df,
            x=group,
            y="accuracy",
            hue="condition",
            palette=[self.cmap(1), self.cmap(0)],
            ax=axes[0],
        )
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0.0, 1.05, 0.1))
        ax.grid(which="both", linewidth=0.5, axis="y", linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xlabel(" ".join(group.title().split("_")))
        ax.set_ylabel("Accuracy")
        ax.set_title(
            f"Accuracy of Direct Answers and Chain-of-Thought per {group.title()}"
        )

        # PLOT TOXIC COT
        plot_df = df.groupby(group)[f"toxic_cot_{version}"].mean().reset_index()
        plot_df = plot_df.melt(group, var_name="version", value_name="toxic_rate")
        ax = sns.barplot(
            data=plot_df,
            x=group,
            y="toxic_rate",
            hue="version",
            palette=[self.cmap(0), self.cmap(1)],
            ax=axes[1],
        )
        ax.set_xlabel(" ".join(group.title().split("_")))
        ax.set_ylabel("Toxic CoT Rate")
        ax.set_title(
            f"Percentage of Toxic Chain-of-Though in Correct Direct Answers per {group.title()}"
        )
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0.0, 1.05, 0.1))
        ax.grid(which="both", linewidth=0.5, axis="y", linestyle="--")
        ax.set_axisbelow(True)
        png_path = self._resolve_save_target(
            f"acc_and_toxic_cot_per_{group}.png",
            Path("/".join(plot_name_add)) if plot_name_add else None,
        )
        self._save_plot(file_name=png_path)

    def plot_acc_two_runs_per(
        self,
        group: str,
        df: pd.DataFrame,
        version: str,
        y_label: str = "Accuracy",
        file_name: str | None = None,
        plot_name_add: list[str] | None = None,
    ) -> None:
        """
        Compare per-task accuracy between runs with and without reasoning.
        """
        plt.close()

        plot_df = (
            df.groupby(group)
            .agg(
                Direct_answer=(f"answer_correct_{version}_da", "mean"),
                With_reasoning=(f"answer_correct_{version}_reas", "mean"),
            )
            .reset_index()
            .melt(id_vars=group, var_name="condition", value_name="accuracy")
        )

        plt.figure(figsize=(len(df[group].unique()) * 0.4, 5))
        ax = sns.barplot(
            data=plot_df,
            x=group,
            y="accuracy",
            hue="condition",
            palette=[self.cmap(1), self.cmap(0)],
        )
        ax.set_title(
            f"Accuracy per {group.title()} for Direct Answers and Chain-of-Thought ({version})"
        )
        ax.set_xlabel(" ".join(group.title().split("_")))
        ax.set_ylabel(y_label)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0.0, 1.05, 0.1))
        plt.grid(which="both", linewidth=0.5, axis="y", linestyle="--")
        plt.gca().set_axisbelow(True)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path_add = Path("/".join(plot_name_add)) if plot_name_add else None
        png_path = self._resolve_save_target(
            file_name or f"{y_label.lower().replace(' ', '_')}_per_{group}.png",
            path_add,
        )
        self._save_plot(file_name=png_path)

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
        path_add = Path("/".join(plot_name_add)) if plot_name_add else None
        if file_name:
            png_path = self._resolve_save_target(file_name, path_add)
            self._save_plot(file_name=png_path)
        else:
            self._save_plot(y_label, x_label, path_add=path_add)

    def plot_correctness_agreement(
        self,
        df: pd.DataFrame,
        versions: list[str],
        plot_name_add: list[str] | None = None,
    ) -> None:
        """
        Compare correctness agreement between runs with and without reasoning.

        :param df: pd.DataFrame
        :param versions: list of versions to compare, e.g. ["before", "after"]
        :param group: optional column name to group by (e.g. "task" or "prompt");
                     if None, agreement is computed across the whole dataset
        :param plot_name_add: optional list of strings to add to the plot name (e.g. for version or group tags)
        :return: None
        """
        plt.close()
        plt.figure(figsize=(8, 6), constrained_layout=True)

        correct_attrs = []
        for version in versions:
            correct_attrs.append(f"answer_correct_{version}_da")
            correct_attrs.append(f"answer_correct_{version}_reas")

        agree = pd.DataFrame(index=correct_attrs, columns=correct_attrs, dtype=float)
        for a in correct_attrs:
            for b in correct_attrs:
                agree.loc[a, b] = (df[a] == df[b]).mean()

        sns.heatmap(agree.astype(float), annot=True, vmin=0, vmax=1, cmap="Blues")
        plt.title(
            f"Answer Correct Agreement Between DA and Reasoning Runs with Versions={versions}"
        )
        plt.xticks(rotation=45, ha="right")
        png_path = self._resolve_save_target(
            f"correctness_agreement_{'_'.join(versions)}.png",
            Path("/".join(plot_name_add)) if plot_name_add else None,
        )
        self._save_plot(file_name=png_path)

    def plot_attr_agreement(
        self,
        df: pd.DataFrame,
        versions: list[str],
        plot_name_add: list[str] | None = None,
    ) -> None:
        """
        Compare correctness agreement between runs with and without reasoning.

        :param df: pd.DataFrame
        :param versions: list of versions to compare, e.g. ["before", "after"]
        :param group: optional column name to group by (e.g. "task" or "prompt");
                     if None, agreement is computed across the whole dataset
        :param plot_name_add: optional list of strings to add to the plot name (e.g. for version or group tags)
        :return: None
        """
        plt.close()
        plt.figure(figsize=(12, 8), constrained_layout=True)
        ATTRS = [
            "max_supp_attn",
            "attn_on_target",
            "there",
            "verbs",
            "pronouns",
            "not_mentioned",
            "context_sents_hall",
        ]
        correct_attrs = []
        for version in versions:
            for attr in ATTRS:
                correct_attrs.append(f"{attr}_{version}_da")
                correct_attrs.append(f"{attr}_{version}_reas")

        agree = pd.DataFrame(index=correct_attrs, columns=correct_attrs, dtype=float)
        for a in correct_attrs:
            for b in correct_attrs:
                agree.loc[a, b] = (df[a] == df[b]).mean()

        sns.heatmap(agree.astype(float), annot=True, vmin=0, vmax=1, cmap="Blues")
        plt.xticks(rotation=45, ha="right")
        plt.title(
            f"Answer Correct Agreement Between DA and Reasoning Runs with Versions={versions}"
        )
        png_path = self._resolve_save_target(
            f"attribute_agreement_{'_'.join(versions)}.png",
            Path("/".join(plot_name_add)) if plot_name_add else None,
        )
        self._save_plot(file_name=png_path)

    def plot_toxic_cot_per(
        self,
        group: str,
        df: pd.DataFrame,
        version: str,
        plot_name_add: list[str] | None = None,
    ) -> None:
        plt.close()
        plot_df = (
            df.groupby(group)[f"toxic_cot_{version}"]
            .mean()
            .reset_index(name="toxic_rate")
        )
        sns.barplot(
            data=plot_df,
            x=group,
            y="toxic_rate",
            color=self.cmap(0),
        )
        # sns.catplot(
        #     data=plot_df,
        #     x=group,
        #     y="toxic_rate",
        #     hue="version",
        #     kind="bar",
        #     height=5,
        #     aspect=1.5,
        # )
        plt.xlabel(" ".join(group.title().split("_")))
        plt.ylabel("Toxic CoT Rate")
        plt.title(
            f"Percentage of Toxic Chain-of-Though in Correct Direct Answers per {group.title()} ({version})"
        )
        png_path = self._resolve_save_target(
            f"toxic_cot_per_{group}.png",
            Path("/".join(plot_name_add)) if plot_name_add else None,
        )
        self._save_plot(file_name=png_path)

    def plot_acc_with_std(
        self,
        acc_per_prompt_task: dict[str | Prompt, Accuracy | Metric],
        x_label: str = "Task",
        y_label: str = "Accuracy",
        file_name=None,
        plot_name_add: list[str] = None,
        path_add: Path = None,
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
        if file_name:
            png_path = self._resolve_save_target(file_name, path_add)
            self._save_plot(file_name=png_path)
        else:
            self._save_plot(y_label, x_label, path_add=path_add)

    def plot_exact_vs_soft_match_per_task(
        self,
        evaluator,
        plot_name_add: list[str] = None,
        path_add: Path = None,
        show_values: bool = False,
    ) -> None:
        """
        Overlay exact-match and soft-match accuracy for a single evaluator,
        one line per metric, plotted over tasks.

        Useful to see at a glance which tasks have a large gap between exact
        and soft match (many partially-correct answers) versus tasks where both
        lines overlap (answers are either fully correct or fully wrong).

        :param evaluator: a MetricEvaluator with exact_match_accuracy and
                          soft_match_accuracy attributes
        :param plot_name_add: extra tags appended to the title
        :param path_add: sub-folder under results_path
        :param show_values: when True, annotate each point with its value
        """
        em = getattr(evaluator, "exact_match_accuracy", None)
        sm = getattr(evaluator, "soft_match_accuracy", None)
        em_std = getattr(evaluator, "exact_match_std", None)
        sm_std = getattr(evaluator, "soft_match_std", None)

        if em is None and sm is None:
            print("[plot_exact_vs_soft_match_per_task] No accuracy data available.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        specs = [
            (em, em_std, self.color_supporting, "Exact match", "o"),
            (sm, sm_std, self.color_distractor, "Soft match", "s"),
        ]
        txt_rows: list[tuple[str, float | None]] = []
        for metric, std_metric, color, label, marker in specs:
            if metric is None:
                continue
            vals = np.array(metric.all if hasattr(metric, "all") else metric)
            x = np.arange(1, len(vals) + 1)
            ax.plot(x, vals, marker=marker, color=color, linewidth=2, label=label)
            if std_metric is not None:
                stds = np.array(
                    std_metric.all if hasattr(std_metric, "all") else std_metric
                )
                if len(stds) == len(vals):
                    ax.fill_between(
                        x, vals - stds, vals + stds, color=color, alpha=0.15
                    )
            for xi, v in zip(x, vals):
                txt_rows.append((f"task={xi} {label}", float(v)))
                if show_values:
                    ax.text(
                        xi,
                        v,
                        f"{v:.2f}",
                        fontsize=7,
                        ha="center",
                        va="bottom",
                        color="#222222",
                    )

        ax.set_xlabel("Task", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(
            np.arange(
                1,
                max(
                    len(em.all if em and hasattr(em, "all") else em or []),
                    len(sm.all if sm and hasattr(sm, "all") else sm or []),
                )
                + 1,
            )
        )
        title = "Exact vs Soft Match Accuracy Per Task"
        if plot_name_add:
            title += f"  ({', '.join(plot_name_add)})"
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()

        png_path = self._resolve_save_target(
            "exact_vs_soft_match_per_task.png", path_add
        )
        self._write_plot_data_txt(png_path, [("Accuracy per task", txt_rows)])
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_reasoning_scores_per_task(
        self,
        evaluator,
        plot_name_add: list[str] = None,
        path_add: Path = None,
        show_values: bool = False,
    ) -> None:
        """
        Line plot of BLEU, ROUGE, and METEOR reasoning scores over tasks for
        a single evaluator.

        All three scores share the y-axis ([0, 1]) and are drawn with distinct
        colours and markers so task-level trends are easy to compare.

        :param evaluator: a MetricEvaluator with bleu, rouge, meteor attributes
        :param plot_name_add: extra tags appended to the title
        :param path_add: sub-folder under results_path
        :param show_values: when True, annotate each point with its value
        """
        score_specs = [
            ("bleu", "bleu_std", self.color_supporting, "BLEU", "o"),
            ("rouge", "rouge_std", self.color_distractor, "ROUGE", "s"),
            ("meteor", "meteor_std", self.color_neutral, "METEOR", "^"),
        ]

        any_data = False
        fig, ax = plt.subplots(figsize=(10, 5))
        txt_rows: list[tuple[str, float | None]] = []

        for attr, std_attr, color, label, marker in score_specs:
            metric = getattr(evaluator, attr, None)
            if metric is None:
                continue
            vals = np.array(metric.all if hasattr(metric, "all") else metric)
            if len(vals) == 0:
                continue
            any_data = True
            x = np.arange(1, len(vals) + 1)
            ax.plot(x, vals, marker=marker, color=color, linewidth=2, label=label)

            std_metric = getattr(evaluator, std_attr, None)
            if std_metric is not None:
                stds = np.array(
                    std_metric.all if hasattr(std_metric, "all") else std_metric
                )
                if len(stds) == len(vals):
                    ax.fill_between(
                        x, vals - stds, vals + stds, color=color, alpha=0.15
                    )

            for xi, v in zip(x, vals):
                txt_rows.append((f"task={xi} {label}", float(v)))
                if show_values:
                    ax.text(
                        xi,
                        v,
                        f"{v:.2f}",
                        fontsize=7,
                        ha="center",
                        va="bottom",
                        color="#222222",
                    )

        if not any_data:
            print("[plot_reasoning_scores_per_task] No reasoning score data available.")
            plt.close(fig)
            return

        ax.set_xlabel("Task", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.05)
        title = "Reasoning scores per task"
        if plot_name_add:
            title += f"  ({', '.join(plot_name_add)})"
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()

        png_path = self._resolve_save_target("reasoning_scores_per_task.png", path_add)
        self._write_plot_data_txt(png_path, [("Reasoning scores per task", txt_rows)])
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_reasoning_vs_direct_answer_per_task(
        self,
        reasoning_evaluator,
        direct_answer_evaluator,
        plot_name_add: list[str] = None,
        path_add: Path = None,
        show_values: bool = False,
    ) -> None:
        """
        Compare exact-match and soft-match accuracy between a reasoning
        evaluator and a direct-answer evaluator, per task.

        Two side-by-side panels (exact match | soft match). Each panel shows
        one line for reasoning and one for direct answer, with std bands,
        making it easy to see where step-by-step reasoning helps or hurts.

        :param reasoning_evaluator: MetricEvaluator from the reasoning experiment
        :param direct_answer_evaluator: MetricEvaluator from the direct-answer
                                        experiment
        :param plot_name_add: extra tags appended to the title
        :param path_add: sub-folder under results_path
        :param show_values: when True, annotate each point with its value
        """
        if reasoning_evaluator is None and direct_answer_evaluator is None:
            print(
                "[plot_reasoning_vs_direct_answer_per_task] "
                "Both evaluators are None; nothing to plot."
            )
            return

        evaluator_specs = [
            (reasoning_evaluator, self.color_supporting, "Reasoning"),
            (direct_answer_evaluator, self.color_distractor, "Direct answer"),
        ]

        metric_specs = [
            ("exact_match_accuracy", "exact_match_std", "Exact match"),
            ("soft_match_accuracy", "soft_match_std", "Soft match"),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        all_rows: list[tuple[str, list[tuple[str, float | None]]]] = []

        for ax, (mean_attr, std_attr, panel_title) in zip(axes, metric_specs):
            txt_rows: list[tuple[str, float | None]] = []
            for evaluator, color, exp_label in evaluator_specs:
                if evaluator is None:
                    continue
                metric = getattr(evaluator, mean_attr, None)
                if metric is None:
                    continue
                vals = np.array(metric.all if hasattr(metric, "all") else metric)
                if len(vals) == 0:
                    continue
                x = np.arange(1, len(vals) + 1)
                ax.plot(x, vals, marker="o", color=color, linewidth=2, label=exp_label)
                std_metric = getattr(evaluator, std_attr, None)
                if std_metric is not None:
                    stds = np.array(
                        std_metric.all if hasattr(std_metric, "all") else std_metric
                    )
                    if len(stds) == len(vals):
                        ax.fill_between(
                            x, vals - stds, vals + stds, color=color, alpha=0.15
                        )
                for xi, v in zip(x, vals):
                    txt_rows.append((f"task={xi} {exp_label}", float(v)))
                    if show_values:
                        ax.text(
                            xi,
                            v,
                            f"{v:.2f}",
                            fontsize=7,
                            ha="center",
                            va="bottom",
                            color="#222222",
                        )

            ax.set_xlabel("Task", fontsize=11)
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.set_ylim(0, 1.05)
            ax.set_title(panel_title, fontsize=11)
            ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            all_rows.append((panel_title, txt_rows))

        title = "Reasoning vs direct answer accuracy per task"
        if plot_name_add:
            title += f"  ({', '.join(plot_name_add)})"
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()

        png_path = self._resolve_save_target(
            "reasoning_vs_direct_answer_per_task.png", path_add
        )
        self._write_plot_data_txt(png_path, all_rows)
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_accuracy_distribution(
        self,
        evaluators: list,
        plot_name_add: list[str] = None,
        path_add: Path = None,
    ) -> None:
        """
        Boxplot (one box per evaluator version) of accuracy values across tasks.

        Shows the spread and median of per-task accuracy in a single glance,
        complementing the per-task line plots with a distributional view.
        Both exact-match and soft-match are shown side by side.

        :param evaluators: list of MetricEvaluator objects (one per version)
        :param plot_name_add: extra tags appended to the title
        :param path_add: sub-folder under results_path
        """
        if not evaluators:
            print("[plot_accuracy_distribution] No evaluators provided.")
            return

        metric_specs = [
            ("exact_match_accuracy", "Exact match"),
            ("soft_match_accuracy", "Soft match"),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        txt_rows: list[tuple[str, float | None]] = []

        for ax, (attr, panel_title) in zip(axes, metric_specs):
            box_data = []
            tick_labels = []
            colors = self.cmap(np.linspace(0, 1, len(evaluators)))

            for evaluator, color in zip(evaluators, colors):
                version = (
                    getattr(evaluator, "version", "")
                    or getattr(evaluator, "name", "")
                    or ""
                )
                metric = getattr(evaluator, attr, None)
                if metric is None:
                    continue
                vals = [
                    v
                    for v in (metric.all if hasattr(metric, "all") else metric)
                    if v is not None and np.isfinite(float(v))
                ]
                if not vals:
                    continue
                box_data.append(vals)
                tick_labels.append(str(version) if version else f"ev{len(box_data)}")
                med = float(np.median(vals))
                txt_rows.append((f"{panel_title} {version} median", med))
                txt_rows.append((f"{panel_title} {version} mean", float(np.mean(vals))))
                txt_rows.append((f"{panel_title} {version} n_tasks", len(vals)))

            if not box_data:
                ax.set_visible(False)
                continue

            bp = ax.boxplot(
                box_data,
                patch_artist=True,
                widths=0.5,
                showmeans=True,
                meanprops=dict(
                    marker="D",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markersize=5,
                ),
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for element in ("whiskers", "caps", "medians"):
                plt.setp(bp[element], color="#333333", linewidth=1.2)

            ax.set_xticks(range(1, len(tick_labels) + 1))
            ax.set_xticklabels(tick_labels, fontsize=10)
            ax.set_ylabel("Accuracy across tasks", fontsize=11)
            ax.set_ylim(0, 1.05)
            ax.set_title(panel_title, fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)

        title = "Accuracy distribution across tasks"
        if plot_name_add:
            title += f"  ({', '.join(plot_name_add)})"
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()

        png_path = self._resolve_save_target("accuracy_distribution.png", path_add)
        self._write_plot_data_txt(png_path, [("Distribution stats", txt_rows)])
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_exact_vs_soft_match_per_task(
        self,
        evaluator,
        plot_name_add: list[str] = None,
        path_add: Path = None,
        show_values: bool = False,
    ) -> None:
        """
        Overlay exact-match and soft-match accuracy for a single evaluator,
        one line per metric, plotted over tasks.

        Useful to see at a glance which tasks have a large gap between exact
        and soft match (many partially-correct answers) versus tasks where both
        lines overlap (answers are either fully correct or fully wrong).

        :param evaluator: a MetricEvaluator with exact_match_accuracy and
                          soft_match_accuracy attributes
        :param plot_name_add: extra tags appended to the title
        :param path_add: sub-folder under results_path
        :param show_values: when True, annotate each point with its value
        """
        em = getattr(evaluator, "exact_match_accuracy", None)
        sm = getattr(evaluator, "soft_match_accuracy", None)
        em_std = getattr(evaluator, "exact_match_std", None)
        sm_std = getattr(evaluator, "soft_match_std", None)

        if em is None and sm is None:
            print("[plot_exact_vs_soft_match_per_task] No accuracy data available.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        specs = [
            (em, em_std, self.color_supporting, "Exact match", "o"),
            (sm, sm_std, self.color_distractor, "Soft match", "s"),
        ]
        txt_rows: list[tuple[str, float | None]] = []
        for metric, std_metric, color, label, marker in specs:
            if metric is None:
                continue
            vals = np.array(metric.all if hasattr(metric, "all") else metric)
            x = np.arange(1, len(vals) + 1)
            ax.plot(x, vals, marker=marker, color=color, linewidth=2, label=label)
            if std_metric is not None:
                stds = np.array(
                    std_metric.all if hasattr(std_metric, "all") else std_metric
                )
                if len(stds) == len(vals):
                    ax.fill_between(
                        x, vals - stds, vals + stds, color=color, alpha=0.15
                    )
            for xi, v in zip(x, vals):
                txt_rows.append((f"task={xi} {label}", float(v)))
                if show_values:
                    ax.text(
                        xi,
                        v,
                        f"{v:.2f}",
                        fontsize=7,
                        ha="center",
                        va="bottom",
                        color="#222222",
                    )

        ax.set_xlabel("Task", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(
            np.arange(
                1,
                max(
                    len(em.all if em and hasattr(em, "all") else em or []),
                    len(sm.all if sm and hasattr(sm, "all") else sm or []),
                )
                + 1,
            )
        )
        title = "Exact vs Soft Match Accuracy Per Task"
        if plot_name_add:
            title += f"  ({', '.join(plot_name_add)})"
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()

        png_path = self._resolve_save_target(
            "exact_vs_soft_match_per_task.png", path_add
        )
        self._write_plot_data_txt(png_path, [("Accuracy per task", txt_rows)])
        self._save_plot(file_name=png_path)
        plt.close(fig)

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
            max_score = (
                max(
                    [
                        val
                        for val in reasoning_scores.values()
                        if not isinstance(val, str)
                    ]
                )
                if reasoning_scores
                else 1.0
            )
            min_score = (
                min(
                    [
                        val
                        for val in reasoning_scores.values()
                        if not isinstance(val, str)
                    ]
                )
                if reasoning_scores
                else 0.0
            )
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
                            try:
                                assert isinstance(score, (int, float))
                            except AssertionError:
                                print(
                                    f"Non-numeric reasoning score for index {idx}: {score}"
                                )
                                warnings.warn(
                                    f"Non-numeric reasoning score for index {idx}, cannot color."
                                )
                                continue
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
                            try:
                                assert isinstance(reasoning_scores[idx], (int, float))
                            except AssertionError:
                                print(
                                    f"Non-numeric reasoning score for index {idx}: {score}"
                                )
                                continue
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
        fig.savefig(out_path, bbox_inches="tight", dpi=300)

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
        self._save_plot(file_name=f"error_case_heatmap_{case_type}.png")

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
            file_name=f"error_histogram_{normalization}_{setting.title().replace(' ', '_')}.png"
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
        self._save_plot(file_name=f"error_case_pie{uniqueness}_{setting}.png")

    def plot_correlation(
        self,
        x_data: dict[str | Prompt, Accuracy | Metric] | list[float] | np.array,
        y_data: list[float],
        x_label: str = "X",
        y_label: str = "Y",
        file_name=None,
        plot_name_add: list[str] = None,
        path_add: Path = None,
        level: str = None,
        include_soft: bool = True,
        label_add: list[str] = [],
        experiment: str = "direct_answer",
        num_samples: int = None,
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
        :param experiment: name of the experiment, e.g. "direct_answer"
        :param num_samples: number of samples considered, for labeling purposes
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
        if isinstance(x_data, dict):
            x_data_points = {
                k: v
                for k, v in x_data.items()
                if include_soft or "soft" not in k.lower()
            }
        else:
            x_data_points = {"data": x_data}
        x_err = (
            [
                x_data_points.pop(k)
                for k in x_data.keys()
                if "std" in k.lower() and k in x_data_points
            ]
            if isinstance(x_data, dict)
            else []
        )
        colors = self.cmap(np.linspace(0, 1, len(x_data_points)), alpha=0.7)

        # Example scaling: avoid zero size, set min/max
        def scale_stddev(stddevs, min_size=20, max_size=300):
            stddevs = np.array(stddevs)
            if stddevs.max() == stddevs.min():
                return np.full_like(stddevs, (min_size + max_size) / 2)
            scaled = (stddevs - stddevs.min()) / (stddevs.max() - stddevs.min())
            return min_size + scaled * (max_size - min_size)

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

            x_vals = metr.all
            y_vals = (
                [y.get_mean() for y in y_data]
                if isinstance(y_data[0], Metric)
                else y_data
            )

            stddev_arr = (
                np.array(std_dev) if std_dev is not None else np.zeros_like(x_vals)
            )
            sizes = (
                scale_stddev(stddev_arr)
                if std_dev is not None
                else np.full_like(x_vals, 50)
            )

            if len(metr) != len(y_data):
                raise ValueError(
                    f"x and y must have the same first dimension, but have shapes {len(metr)} and {len(y_data)}"
                )

            if not y_data:
                raise ValueError("y_data is empty")

            plt.scatter(
                x_vals,
                y_vals,
                s=sizes,
                # xerr=stddev_arr,
                label=(
                    "{}{}".format(
                        " ".join(metr_type.split("_")).title(),
                        "\nwith Std Dev" if x_err else "",
                    )
                    if isinstance(metr_type, str)
                    else metr_type.name
                ),
                color=color,
                edgecolors="black",
                zorder=3,
            )

            seen_points = dict()
            for i, label in enumerate(label_add):
                if label in seen_points.values():
                    continue  # skip if we've already labeled this point
                x, y = metr[i], (
                    y_data[i].get_mean() if isinstance(y_data[i], Metric) else y_data[i]
                )
                # Find all indices with the same x and y values (within a small tolerance to account for floating point issues)
                same_points = [
                    j
                    for j in range(len(metr))
                    if abs(metr[j] - x) < 1e-6
                    and abs(
                        (
                            y_data[j].get_mean()
                            if isinstance(y_data[j], Metric)
                            else y_data[j]
                        )
                        - y
                    )
                    < 1e-6
                ]
                # Skip points we've already labeled
                same_points = [
                    j
                    for j in same_points
                    if j not in seen_points and label_add[j] != label
                ]  # also check that the label is different to avoid labeling the same point multiple times if it has the same label
                seen_points.update({j: label_add[j] for j in same_points})
                # Summarize the labels for these points (e.g. if they differ only by prompt, we can just list the prompts)
                if len(same_points) >= 1:
                    same_labels = [label_add[j] for j in same_points]
                    assert (
                        label not in same_labels
                    ), "The label for the current point should not be in the same_labels list"
                    summarized_label = f"{label} ({', '.join(same_labels)})"
                else:
                    summarized_label = label
                plt.annotate(
                    summarized_label,
                    (metr[i] + 0.001, y_data[i] + 0.001),
                    xytext=(5, 5 if i % 2 == 0 else -5),
                    textcoords="offset points",
                )

        idx_min, idx_max = np.argmin(stddev_arr), np.argmax(stddev_arr)
        # idx_median = np.argsort(stddev_arr)[len(stddev_arr) // 2]
        median_val = np.median(stddev_arr)
        idx_median = np.argmin(np.abs(stddev_arr - median_val))

        legend_handles = [
            Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=np.sqrt(size),  # markersize is diameter in points
                label=f"Std Dev: {val:.2f}",
            )
            for val, size in set(
                zip(
                    [stddev_arr[idx_min], stddev_arr[idx_median], stddev_arr[idx_max]],
                    sizes[[idx_min, idx_median, idx_max]],
                )
            )
        ]
        legend_handles = sorted(
            legend_handles,
            key=lambda h: float(h.get_label().split(": ")[1]),
            reverse=True,
        )

        # 3. Get existing handles/labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # 4. Combine and set the legend
        combined_handles = handles + legend_handles
        combined_labels = labels + [h.get_label() for h in legend_handles]

        self._plot_general_details(
            x_label,
            y_label,
            max_x_len,
            plot_name_add,
            num_of_data_arrays=num_of_data_arrays,
            metr_types=metr_types,
            step=0.1 if max_x_len == 1 else step_size,
            min_x_len=min_x_len,
            combined_handles=combined_handles,
            combined_labels=combined_labels,
            legend_title="Metric Size & Circle Size",
            experiment=experiment,
            # num_parts=len(x_data) if isinstance(x_data, (list, np.ndarray)) else None, # This may be confusing, better to stay with samples
            num_samples=num_samples,
        )
        if file_name:
            png_path = self._resolve_save_target(str(file_name).lower(), path_add)
        else:
            label = y_label.lower().replace(" ", "_")
            png_path = self._resolve_save_target(
                f"{label}_per_{x_label.lower()}.png", path_add
            )
        self._save_plot(file_name=png_path)
        plt.close()

    def plot_corr_hist(
        self,
        x_data: dict[str | Prompt, Accuracy | Metric],
        y_data: dict[str, list[float] | np.array] = None,
        x_label: str = "X",
        y_label: str = "Y",
        displ_percentage: bool = False,
        file_name: str = None,
        plot_name_add: list[str] = None,
        level: str = None,
        id: int = 1,
        path_add: Path = None,
        experiment: str = "direct_answer",
        num_samples: int = None,
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
        :param experiment: name of the experiment, e.g. "direct_answer"
        :param num_samples: number of samples considered, for labeling purposes
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

        x_data = {x_label: x_data} if isinstance(x_data, (list, np.ndarray)) else x_data

        df = pd.DataFrame(
            list(zip(*x_data.values(), *df_data.values())),
            columns=[x_label] + list(df_data.keys()),
        )
        max_x_len = max(df[x_label])
        min_x_len = min(df[x_label])
        if max_x_len > 100:
            step_size = 5
        elif max_x_len > 30:
            step_size = 2
        else:
            step_size = 1

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
        step_size = 4 if max_x_len >= 100 else 2 if max_x_len >= 30 else 1

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
            min_x_len=min_x_len,
            num_of_data_arrays=1,
            displ_percentage=displ_percentage,
            plot_name_add=plot_name_add,
            legend_title=label_column,
            step=step_size,
            experiment=experiment,
            num_samples=num_samples,
        )

        png_path = self._resolve_save_target(file_name.lower(), path_add)
        self._write_plot_data_txt(
            png_path,
            [("Plot data", [(str(x_label), None)])],
        )
        self._save_plot(file_name=png_path)
        plt.close()

    def plot_corr_boxplot(
        self,
        x_data: dict[str | Prompt, Accuracy | Metric] | list[float] | np.array,
        y_data: dict[str : list[float] | np.array] = None,
        x_label: str = "X",
        y_label: str = "Y",
        displ_percentage: bool = False,
        version: str = False,
        file_name: str = None,
        plot_name_add: list[str] = None,
        path_add: Path = None,
        level: str = None,
        experiment: str = None,
        num_samples: int = None,
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
        :param experiment: name of the experiment, e.g. "direct_answer"
        :param num_samples: number of samples considered, for labeling purposes
        :return: None
        """
        # Part-level if x_data is list/array, else sample/task-level if dict
        # Currently only used for part-level plots

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
        x_data_points = (
            {x_label: x_data} if isinstance(x_data, (list, np.ndarray)) else x_data
        )
        df = pd.DataFrame(
            list(zip(*x_data_points.values(), *df_data.values())),
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
            feat_str = [f.removesuffix(f"_{version}") for f in feat_str]
            return "-".join(feat_str) if feat_str else None

        if any(lab in x_label.lower() for lab in ["correct", "in self"]):
            df[x_label] = df[x_label].map(
                {
                    0: (
                        "In previous parts"
                        if "in self" in x_label.lower()
                        else "Incorrect"
                    ),
                    1: "In current part" if "in self" in x_label.lower() else "Correct",
                }
            )
        elif "target" in x_label.lower():
            df[x_label] = df[x_label].astype(int)
        else:
            df[x_label] = df[x_label].round()
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
        hue_col = f"{label_column}_"
        use_hue = (
            hue_col in df.columns and df[hue_col].nunique(dropna=True) >= 1
        )  # only use hue if there is at least one non-NaN value

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
        ax.set_axisbelow(True)

        self._plot_general_details(
            x_label=x_label,
            y_label=y_label,
            max_x_len=len(x_data),
            num_of_data_arrays=1,
            displ_percentage=displ_percentage,
            plot_name_add=plot_name_add,
            legend_title=" ".join(label_column.split("_")).title(),
            experiment=experiment,
            num_samples=num_samples,
            num_parts=len(x_data) if isinstance(x_data, (list, np.ndarray)) else None,
        )

        png_path = self._resolve_save_target(file_name.lower(), path_add)
        self._write_plot_data_txt(
            png_path,
            [("Plot data", [(str(x_label), None)])],
        )
        self._save_plot(file_name=png_path)
        plt.close()

    def plot_distractor_attn_boxplot(
        self,
        stats: DistractorAttentionStats,
        version: str,
        plot_name_add: list[str] = None,
        path_add: Path = None,
        show_values: bool = False,
    ) -> None:
        """
        Boxplot of mean attention on distractor vs neutral context, split by
        answer correctness.

        Uses the full context up to each question (including previous parts).
        """
        grouped = stats.as_grouped()

        role_color = {
            "distractor": self.color_distractor,
            "neutral": self.color_neutral,
        }
        ordering = [
            (True, "distractor"),
            (True, "neutral"),
            (False, "distractor"),
            (False, "neutral"),
        ]
        label_correct = {True: "Correct", False: "Incorrect"}

        box_data, tick_labels, colors, medians, ns = [], [], [], [], []
        # tick_labels [1,2] = Correct group, [4,5] = Incorrect group; gap at 3
        group_ordering = [
            (True, "distractor", 1),
            (True, "neutral", 2),
            (False, "distractor", 4),
            (False, "neutral", 5),
        ]
        for correct, role, pos in group_ordering:
            vals = grouped[correct].get(role, [])
            if vals:
                box_data.append(vals)
                tick_labels.append(pos)
                colors.append(role_color[role])
                medians.append(float(np.median(vals)))
                ns.append(len(vals))

        if not box_data:
            print(
                f"[plot_distractor_attn_boxplot] No data to plot for version='{version}'."
            )
            return

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bp = ax.boxplot(box_data, positions=tick_labels, patch_artist=True, widths=0.55)

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
        for element in ("whiskers", "caps", "fliers", "medians"):
            plt.setp(bp[element], color="#333333", linewidth=1.1)

        # Correctness group labels at group centres; vertical divider between groups.
        ax.set_xticks([1.5, 4.5])
        ax.set_xticklabels(["Correct", "Incorrect"], fontsize=10)
        ax.axvline(3, color="#aaaaaa", linestyle=":", linewidth=0.8)
        ax.set_xlim(0, 6)

        # Colour legend maps sentence role to colour (correctness is already on x-axis).
        from matplotlib.patches import Patch as _Patch

        ax.legend(
            handles=[
                _Patch(
                    facecolor=role_color["distractor"], alpha=0.78, label="Distractor"
                ),
                _Patch(facecolor=role_color["neutral"], alpha=0.78, label="Neutral"),
            ],
            fontsize=9,
            loc="upper right",
            framealpha=0.9,
        )
        ax.set_ylabel("Mean attention", fontsize=11)
        ax.set_ylim(*attn_ylim(*box_data))
        ax.set_title(
            self._format_short_title(
                "Attention on distractor vs neutral", plot_name_add
            ),
            fontsize=11,
            pad=8,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        if show_values:
            for i, m in enumerate(medians, 1):
                ax.text(
                    i,
                    m,
                    f"{m:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#222222",
                )

        fig.tight_layout()

        png_path = self._resolve_save_target(
            f"distractor_attn_boxplot_{version}.png", path_add
        )
        self._write_plot_data_txt(
            png_path,
            [
                (
                    "Box medians by group",
                    [(label, m) for label, m in zip(tick_labels, medians)],
                ),
                (
                    "Sample counts (n)",
                    [(label, n) for label, n in zip(tick_labels, ns)],
                ),
            ],
        )
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_distractor_attn_per_task(
        self,
        stats: DistractorAttentionStats,
        version: str,
        plot_name_add: list[str] = None,
        path_add: Path = None,
        show_values: bool = False,
    ) -> None:
        """
        Grouped bar chart of mean distractor and neutral attention per task,
        with correct and incorrect answers shown side by side.

        :param stats: accumulated distractor attention records for this version
        :param version: "before" or "after"; appears in the filename
        :param plot_name_add: extra tags appended to the title (in brackets)
        :param path_add: sub-folder under results_path
        :param show_values: when True, write each bar's height above it
        """
        per_task = stats.as_per_task()
        task_ids = sorted(per_task.keys())
        if not task_ids:
            print(
                f"[plot_distractor_attn_per_task] No data to plot for version='{version}'."
            )
            return

        # Compute per-task supporting means directly from records (as_per_task may
        # only expose distractor and neutral).
        _supp: dict = defaultdict(lambda: {True: [], False: []})
        for r in stats.records:
            if r.attn_supporting is not None:
                _supp[r.task_id][bool(r.answer_correct)].append(r.attn_supporting)
        supp_per_task = {
            tid: {c: float(np.mean(vs)) if vs else None for c, vs in by_c.items()}
            for tid, by_c in _supp.items()
        }

        x = np.arange(len(task_ids))
        width = 0.13

        # (correct, role, offset_mult, color, label, hatch)
        # Solid fill = correct; hatched = incorrect for unambiguous differentiation.
        bar_spec = [
            (
                True,
                "distractor",
                -2.5,
                self.color_distractor,
                "Correct / distractor",
                "",
            ),
            (
                True,
                "supporting",
                -1.5,
                self.color_supporting,
                "Correct / supporting",
                "",
            ),
            (
                True,
                "neutral",
                -0.5,
                self.color_neutral,
                "Correct / neutral",
                "",
            ),
            (
                False,
                "distractor",
                0.5,
                self.color_distractor,
                "Incorrect / distractor",
                "//",
            ),
            (
                False,
                "supporting",
                1.5,
                self.color_supporting,
                "Incorrect / supporting",
                "//",
            ),
            (
                False,
                "neutral",
                2.5,
                self.color_neutral,
                "Incorrect / neutral",
                "//",
            ),
        ]

        fig, ax = plt.subplots(figsize=(max(7, len(task_ids) * 0.9), 4.5))

        txt_rows: list[tuple[str, float | None]] = []
        try:

            for correct, role, offset_mult, color, label, hatch in bar_spec:
                if role == "supporting":
                    means = [supp_per_task.get(tid, {}).get(correct) for tid in task_ids]
                else:
                    means = [
                        per_task[tid].get(correct, {}).get(role, None) for tid in task_ids
                    ]
                xs, heights = [], []
                for i, m in enumerate(means):
                    if m is not None:
                        xs.append(x[i] + offset_mult * width)
                        heights.append(m)
                    txt_rows.append((f"task={task_ids[i]} {label}", m))
                if xs:
                    bars = ax.bar(
                        xs,
                        heights,
                        width,
                        label=label,
                        color=color,
                        alpha=0.82,
                        hatch=hatch,
                    )
                    if show_values:
                        for b, h in zip(bars, heights):
                            ax.text(
                                b.get_x() + b.get_width() / 2,
                                h,
                                f"{h:.2f}",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                                color="#222222",
                            )
        except ValueError as e:
            print(str(e)) # Unexpected number of returned bar_spec variables
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{tid}" for tid in task_ids], fontsize=9)
        ax.set_ylabel("Mean attention", fontsize=11)
        ax.set_ylim(
            *attn_ylim(
                *[
                    [per_task[tid].get(c, {}).get(r) for tid in task_ids]
                    for c in (True, False)
                    for r in ("distractor", "neutral")
                ],
                *[
                    [supp_per_task.get(tid, {}).get(c) for tid in task_ids]
                    for c in (True, False)
                ],
            )
        )
        ax.set_title(
            self._format_short_title(
                "Attention by sentence role and correctness per task", plot_name_add
            ),
            fontsize=11,
            pad=8,
        )
        ax.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()

        png_path = self._resolve_save_target(
            f"distractor_attn_per_task_{version}.png", path_add
        )
        self._write_plot_data_txt(png_path, [("Bar means", txt_rows)])
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_distractor_attn_scatter(
        self,
        stats: DistractorAttentionStats,
        version: str,
        plot_name_add: list[str] = None,
        path_add: Path = None,
        show_values: bool = False,
    ) -> None:
        """
        Per-part scatter of distractor attention vs neutral attention, coloured
        by answer correctness. The y=x reference line marks where distractor
        and neutral attention are equal.

        :param stats: accumulated distractor attention records for this version
        :param version: "before" or "after"; appears in the filename
        :param plot_name_add: extra tags appended to the title (in brackets)
        :param path_add: sub-folder under results_path
        :param show_values: when True, label group means on the plot
        """
        scatter_data = stats.as_scatter_data()

        # Compute supporting vs neutral from raw records for the second panel.
        supp_scatter: dict = {
            True: {"supporting": [], "neutral": []},
            False: {"supporting": [], "neutral": []},
        }
        for r in stats.records:
            if r.attn_supporting is not None and r.attn_neutral is not None:
                supp_scatter[bool(r.answer_correct)]["supporting"].append(
                    r.attn_supporting
                )
                supp_scatter[bool(r.answer_correct)]["neutral"].append(r.attn_neutral)

        fig, (ax_dist, ax_supp) = plt.subplots(1, 2, figsize=(11, 5.5))

        plot_spec = [
            (True, self.color_correct, "Correct", "o"),
            (False, self.color_incorrect, "Incorrect", "^"),
        ]
        any_data = False
        txt_rows: list[tuple[str, float | int | None]] = []

        # --- left panel: distractor vs neutral (existing) ---
        for correct, color, label, marker in plot_spec:
            xvals = scatter_data[correct]["neutral"]
            yvals = scatter_data[correct]["distractor"]
            if xvals:
                any_data = True
                ax_dist.scatter(
                    xvals,
                    yvals,
                    alpha=0.55,
                    s=32,
                    color=color,
                    marker=marker,
                    label=f"{label} (n={len(xvals)})",
                    edgecolors="none",
                )
                mx, my = float(np.mean(xvals)), float(np.mean(yvals))
                txt_rows.append((f"{label} distractor_vs_neutral n", len(xvals)))
                txt_rows.append((f"{label} mean(neutral)", mx))
                txt_rows.append((f"{label} mean(distractor)", my))
                if show_values:
                    ax_dist.text(
                        mx, my, f"  ({mx:.2f},{my:.2f})", fontsize=8, color="#222222"
                    )

        # --- right panel: supporting vs neutral ---
        any_supp = False
        for correct, color, label, marker in plot_spec:
            xvals = supp_scatter[correct]["neutral"]
            yvals = supp_scatter[correct]["supporting"]
            if xvals:
                any_supp = True
                ax_supp.scatter(
                    xvals,
                    yvals,
                    alpha=0.55,
                    s=32,
                    color=color,
                    marker=marker,
                    label=f"{label} (n={len(xvals)})",
                    edgecolors="none",
                )
                mx, my = float(np.mean(xvals)), float(np.mean(yvals))
                txt_rows.append((f"{label} supporting_vs_neutral n", len(xvals)))
                txt_rows.append((f"{label} mean(neutral)", mx))
                txt_rows.append((f"{label} mean(supporting)", my))
                if show_values:
                    ax_supp.text(
                        mx, my, f"  ({mx:.2f},{my:.2f})", fontsize=8, color="#222222"
                    )

        if not any_data and not any_supp:
            print(
                f"[plot_distractor_attn_scatter] No data to plot for version='{version}'."
            )
            plt.close(fig)
            return

        def _style_scatter_ax(ax, xlabel, ylabel, title, data_groups):
            """Apply equal-axis limits and reference line."""
            all_vals = []
            for group in data_groups:
                all_vals.extend(group)
            _lo, _ceil = attn_ylim(all_vals)
            ax.plot(
                [_lo, _ceil],
                [_lo, _ceil],
                "k--",
                linewidth=0.9,
                alpha=0.45,
                label="y = x",
            )
            ax.set_xlim(_lo, _ceil)
            ax.set_ylim(_lo, _ceil)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(
                self._format_short_title(title, plot_name_add), fontsize=11, pad=8
            )
            ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
            ax.grid(linestyle="--", alpha=0.35)
            ax.set_axisbelow(True)

        _style_scatter_ax(
            ax_dist,
            "Neutral attention",
            "Distractor attention",
            "Per-part distractor vs neutral",
            [
                scatter_data[c]["neutral"] + scatter_data[c]["distractor"]
                for c in (True, False)
            ],
        )
        _style_scatter_ax(
            ax_supp,
            "Neutral attention",
            "Supporting attention",
            "Per-part supporting vs neutral",
            [
                supp_scatter[c]["neutral"] + supp_scatter[c]["supporting"]
                for c in (True, False)
            ],
        )

        fig.tight_layout()

        png_path = self._resolve_save_target(
            f"distractor_attn_scatter_{version}.png", path_add
        )
        self._write_plot_data_txt(png_path, [("Group statistics", txt_rows)])
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_supporting_attention(
        self,
        stats: DistractorAttentionStats,
        plot_name_add: list[str] | None = None,
        version: str = "before",
        path_add: Path | None = None,
        show_values: bool = False,
    ) -> None:
        """
        Per-sample distraction margin (attention on distractor minus attention
        on supporting), plotted as a strip plot split by answer correctness.

        Margin > 0 → model is being distracted on that part.
        Margin < 0 → model stays focused on the supporting facts.

        :param stats: accumulated distractor attention records for this version
        :param plot_name_add: extra tags appended to the title (in brackets)
        :param version: "before" or "after"; appears in the filename
        :param path_add: sub-folder under results_path
        :param show_values: when True, label the group means and "% distracted"
        """
        margins_correct: list[float] = []
        margins_incorrect: list[float] = []
        for r in stats.records:
            if r.attn_distractor is None or r.attn_supporting is None:
                continue
            margin = r.attn_distractor - r.attn_supporting
            (margins_correct if r.answer_correct else margins_incorrect).append(margin)

        if not margins_correct and not margins_incorrect:
            print(
                f"[plot_supporting_attention] No data to plot for version='{version}'."
            )
            return

        groups = [
            ("Correct", margins_correct, self.color_correct),
            ("Incorrect", margins_incorrect, self.color_incorrect),
        ]

        fig, ax = plt.subplots(figsize=(6.5, 5))

        rng = np.random.default_rng(seed=0)
        means = []
        pct_distracted = []
        for i, (_, vals, color) in enumerate(groups):
            if not vals:
                means.append(None)
                pct_distracted.append(None)
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                alpha=0.55,
                s=26,
                color=color,
                edgecolors="none",
            )
            m = float(np.mean(vals))
            means.append(m)
            pct = 100.0 * sum(v > 0 for v in vals) / len(vals)
            pct_distracted.append(pct)
            ax.scatter(
                [i],
                [m],
                marker="D",
                s=70,
                color="black",
                zorder=5,
                label="Group mean" if i == 0 else None,
            )
            if show_values:
                ax.text(
                    i + 0.05,
                    m,
                    f"{m:+.2f}",
                    fontsize=9,
                    color="#111111",
                    va="center",
                )

        # Reference line: margin = 0.
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        _mlo, _mceil = margin_ylim(margins_correct, margins_incorrect)
        ax.axhspan(0, _mceil, facecolor=self.color_incorrect, alpha=0.05, zorder=0)
        ax.axhspan(_mlo, 0, facecolor=self.color_correct, alpha=0.05, zorder=0)

        # Brief region cues at the y-axis edge — far less text than before.
        ax.set_ylabel("Distractor − supporting attention", fontsize=11)
        ax.set_xlabel("Answer", fontsize=11)
        ax.set_ylim(_mlo, _mceil)

        # Concise tick labels: just the group name and n.
        xtick_labels = []
        for label, vals, _ in groups:
            xtick_labels.append(f"{label}\nn = {len(vals)}")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(xtick_labels, fontsize=10)

        ax.set_title(
            self._format_short_title("Distraction margin per part", plot_name_add),
            fontsize=11,
            pad=8,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        fig.tight_layout()

        png_path = self._resolve_save_target(
            f"supporting_attention_{version}.png", path_add
        )
        txt_rows: list[tuple[str, float | int | None]] = []
        for (label, vals, _), m, pct in zip(groups, means, pct_distracted):
            txt_rows.append((f"{label} n", len(vals)))
            txt_rows.append((f"{label} mean margin", m))
            txt_rows.append((f"{label} pct distracted", pct))
        self._write_plot_data_txt(png_path, [("Group statistics", txt_rows)])
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_distractor_supporting_ratio(
        self,
        stats: DistractorAttentionStats,
        plot_name_add: list[str] | None = None,
        eps: float = 1e-8,
        version: str = "before",
        path_add: Path | None = None,
        show_values: bool = False,
    ) -> None:
        """
        Per-sample boxplot of the distractor / supporting attention ratio,
        on a log scale, split by answer correctness.

        Ratio > 1 → distractor receives more attention than supporting.
        Ratio < 1 → supporting still wins.

        :param stats: accumulated distractor attention records for this version
        :param plot_name_add: extra tags appended to the title (in brackets)
        :param eps: stabiliser for division
        :param version: "before" or "after"; appears in the filename
        :param path_add: sub-folder under results_path
        :param show_values: when True, label group medians and "% above 1"
        """
        ratios_correct: list[float] = []
        ratios_incorrect: list[float] = []
        for r in stats.records:
            if r.attn_supporting is None or r.attn_distractor is None:
                continue
            ratio = r.attn_distractor / (r.attn_supporting + eps)
            if ratio <= 0:
                continue
            (ratios_correct if r.answer_correct else ratios_incorrect).append(ratio)

        if not ratios_correct and not ratios_incorrect:
            print(
                f"[plot_distractor_supporting_ratio] No data to plot for "
                f"version='{version}'."
            )
            return

        groups = [
            ("Correct", ratios_correct, self.color_correct),
            ("Incorrect", ratios_incorrect, self.color_incorrect),
        ]

        fig, ax = plt.subplots(figsize=(6.5, 5))

        # Boxplot needs a non-empty list per group; substitute NaN if absent.
        box_data = [vals if vals else [np.nan] for _, vals, _ in groups]
        bp = ax.boxplot(
            box_data,
            patch_artist=True,
            widths=0.5,
            showmeans=True,
            meanprops=dict(
                marker="D",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=6,
            ),
        )
        for patch, (_, _, color) in zip(bp["boxes"], groups):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for element in ("whiskers", "caps", "medians"):
            plt.setp(bp[element], color="#333333", linewidth=1.1)

        # Reference line at ratio = 1.0 and shaded half-planes.
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_yscale("log")

        # Data-driven limits: pad one half-decade beyond the observed range so
        # whiskers are never clipped, but don't waste space on orders of magnitude
        # with no data.  Fall back to the global constants if data is absent.
        all_ratios = ratios_correct + ratios_incorrect
        if all_ratios:
            log_vals = np.array([np.log10(r) for r in all_ratios if r > 0])
            lo_log = max(np.log10(RATIO_YMIN), log_vals.min() - 0.5)
            hi_log = min(np.log10(RATIO_YMAX), log_vals.max() + 0.5)
        else:
            lo_log, hi_log = np.log10(RATIO_YMIN), np.log10(RATIO_YMAX)
        ax.set_ylim(10**lo_log, 10**hi_log)

        ax.axhspan(
            1.0, 10**hi_log, facecolor=self.color_incorrect, alpha=0.05, zorder=0
        )
        ax.axhspan(10**lo_log, 1.0, facecolor=self.color_correct, alpha=0.05, zorder=0)

        # Compact tick labels with just count and percentages.
        medians: list[float | None] = []
        pct_above: list[float | None] = []
        xtick_labels: list[str] = []
        for label, vals, _ in groups:
            if vals:
                medians.append(float(np.median(vals)))
                pct = 100.0 * sum(v > 1.0 for v in vals) / len(vals)
                pct_above.append(pct)
                xtick_labels.append(f"{label}\nn = {len(vals)}")
            else:
                medians.append(None)
                pct_above.append(None)
                xtick_labels.append(f"{label}\nn = 0")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(xtick_labels, fontsize=10)
        ax.set_ylabel(
            "Distractor attention / supporting attention (log scale)", fontsize=11
        )
        ax.set_xlabel("Answer", fontsize=11)
        ax.set_title(
            self._format_short_title("Distractor-to-supporting ratio", plot_name_add),
            fontsize=11,
            pad=8,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4, which="both")
        ax.set_axisbelow(True)

        if show_values:
            for i, (m, pct) in enumerate(zip(medians, pct_above), 1):
                if m is not None:
                    ax.text(
                        i + 0.06,
                        m,
                        f"med={m:.2f}\n>1: {pct:.0f}%",
                        fontsize=8,
                        color="#222222",
                        va="center",
                    )

        fig.tight_layout()

        png_path = self._resolve_save_target(
            f"distractor_supporting_ratio_{version}.png", path_add
        )
        txt_rows: list[tuple[str, float | int | None]] = []
        for (label, vals, _), m, pct in zip(groups, medians, pct_above):
            txt_rows.append((f"{label} n", len(vals)))
            txt_rows.append((f"{label} median ratio", m))
            txt_rows.append((f"{label} pct above 1", pct))
        self._write_plot_data_txt(png_path, [("Group statistics", txt_rows)])
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_attention_triplet(
        self,
        stats: DistractorAttentionStats,
        plot_name_add: list[str] | None = None,
        version: str = "before",
        path_add: Path | None = None,
        show_values: bool = False,
    ) -> None:
        """
        Bar chart of mean attention on supporting, distractor, and neutral
        sentences, with correct/incorrect side by side. Error bars show SEM.

        :param stats: accumulated distractor attention records for this version
        :param plot_name_add: extra tags appended to the title (in brackets)
        :param version: "before" or "after"; appears in the filename
        :param path_add: sub-folder under results_path
        :param show_values: when True, write each bar's value above it
        """
        data = extract_attention_by_correct(stats)

        categories = ["supporting", "distractor", "neutral"]
        category_labels = ["Supporting", "Distractor", "Neutral"]
        x = np.arange(len(categories))
        width = 0.36

        def _mean_sem_n(vals: list) -> tuple[float, float, int]:
            clean = [v for v in vals if v is not None]
            if not clean:
                return float("nan"), 0.0, 0
            arr = np.asarray(clean, dtype=float)
            sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            return float(arr.mean()), float(sem), len(arr)

        stats_correct = [_mean_sem_n(data[True][c]) for c in categories]
        stats_incorrect = [_mean_sem_n(data[False][c]) for c in categories]

        means_correct = [safe_mean(data[True][c]) for c in categories]
        sems_correct = [s[1] for s in stats_correct]
        ns_correct = [s[2] for s in stats_correct]
        means_incorrect = [safe_mean(data[False][c]) for c in categories]
        sems_incorrect = [s[1] for s in stats_incorrect]
        ns_incorrect = [s[2] for s in stats_incorrect]

        fig, ax = plt.subplots(figsize=(7.5, 5))

        bars_c = ax.bar(
            x - width / 2,
            means_correct,
            width,
            yerr=sems_correct,
            capsize=4,
            label=f"Correct (n = {max(ns_correct) if ns_correct else 0})",
            color=self.color_correct,
            alpha=0.82,
            edgecolor="#1e8449",
        )
        bars_i = ax.bar(
            x + width / 2,
            means_incorrect,
            width,
            yerr=sems_incorrect,
            capsize=4,
            label=f"Incorrect (n = {max(ns_incorrect) if ns_incorrect else 0})",
            color=self.color_incorrect,
            alpha=0.82,
            edgecolor="#a93226",
        )

        if show_values:

            def _annotate(bars, means, sems):
                for bar, m, s in zip(bars, means, sems):
                    if not np.isfinite(m):
                        continue
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        m + s + 0.005,
                        f"{m:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="#222222",
                    )

            _annotate(bars_c, means_correct, sems_correct)
            _annotate(bars_i, means_incorrect, sems_incorrect)

        ax.set_xticks(x)
        ax.set_xticklabels(category_labels, fontsize=10)
        ax.set_xlabel("Sentence role", fontsize=11)
        ax.set_ylabel("Mean attention (± SEM)", fontsize=11)
        ax.set_ylim(*attn_ylim(means_correct, means_incorrect))
        ax.set_title(
            self._format_short_title("Attention by sentence role", plot_name_add),
            fontsize=11,
            pad=8,
        )
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()

        png_path = self._resolve_save_target(
            f"attention_triplet_{version}.png", path_add
        )
        txt_rows = []
        for cat, (mc, sc, nc), (mi, si, ni) in zip(
            category_labels, stats_correct, stats_incorrect
        ):
            txt_rows.append((f"{cat} correct mean", mc))
            txt_rows.append((f"{cat} correct sem", sc))
            txt_rows.append((f"{cat} correct n", nc))
            txt_rows.append((f"{cat} incorrect mean", mi))
            txt_rows.append((f"{cat} incorrect sem", si))
            txt_rows.append((f"{cat} incorrect n", ni))
        self._write_plot_data_txt(
            png_path, [("Means by role and correctness", txt_rows)]
        )
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_distraction_vs_n_distractors(
        self,
        stats: DistractorAttentionStats,
        plot_name_add: list[str] | None = None,
        version: str = "before",
        min_bin_size: int = 3,
        path_add: Path | None = None,
        show_values: bool = False,
    ) -> None:
        """
        Mean distractor and supporting attention plus accuracy as a function
        of the number of distractor sentences in the prompt. Attention uses
        the left y-axis; accuracy uses the right.

        :param stats: accumulated distractor attention records for this version
        :param plot_name_add: extra tags appended to the title (in brackets)
        :param version: "before" or "after"; appears in the filename
        :param min_bin_size: minimum number of records per bin to plot it
        :param path_add: sub-folder under results_path
        :param show_values: when True, label each accuracy point with its value
        """
        by_n: dict[int, list] = defaultdict(list)
        for r in stats.records:
            if r.attn_supporting is None:
                continue
            if r.attn_distractor is None and r.n_distractors != 0:
                continue
            by_n[r.n_distractors].append(r)

        ns = sorted(k for k, v in by_n.items() if len(v) >= min_bin_size)
        if not ns:
            print(
                f"[plot_distraction_vs_n_distractors] No bins meet "
                f"min_bin_size={min_bin_size} for version='{version}'."
            )
            return

        def _mean_sem(vals: list[float]) -> tuple[float, float]:
            arr = np.asarray(
                [v for v in vals if v is not None and np.isfinite(v)], dtype=float
            )
            if arr.size == 0:
                return float("nan"), 0.0
            sem = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
            return float(arr.mean()), float(sem)

        def _normal_ci(k: int, n: int, z: float = 1.96):
            if n == 0:
                return float("nan"), float("nan"), float("nan")
            p = k / n
            se = np.sqrt(p * (1 - p) / n)
            return p, max(0.0, p - z * se), min(1.0, p + z * se)

        dist_mean, dist_sem = [], []
        supp_mean, supp_sem = [], []
        acc, acc_lo, acc_hi = [], [], []
        bin_n = []
        for n in ns:
            recs = by_n[n]
            dist_vals = [
                (
                    0.0
                    if (r.attn_distractor is None and r.n_distractors == 0)
                    else r.attn_distractor
                )
                for r in recs
                if (r.attn_distractor is not None or r.n_distractors == 0)
            ]
            m, s = _mean_sem(dist_vals)
            dist_mean.append(m)
            dist_sem.append(s)
            m, s = _mean_sem([r.attn_supporting for r in recs])
            supp_mean.append(m)
            supp_sem.append(s)
            k = sum(1 for r in recs if r.answer_correct)
            p, lo, hi = _normal_ci(k, len(recs))
            acc.append(p)
            acc_lo.append(lo)
            acc_hi.append(hi)
            bin_n.append(len(recs))

        fig, ax_attn = plt.subplots(figsize=(7.5, 5))
        ax_acc = ax_attn.twinx()

        line_d = ax_attn.plot(
            ns,
            dist_mean,
            marker="o",
            color=self.color_distractor,
            linewidth=2,
            label="Distractor attention",
            linestyle="--",
            alpha=0.75,
        )[0]
        ax_attn.fill_between(
            ns,
            np.asarray(dist_mean) - np.asarray(dist_sem),
            np.asarray(dist_mean) + np.asarray(dist_sem),
            color=self.color_distractor,
            alpha=0.15,
        )
        line_s = ax_attn.plot(
            ns,
            supp_mean,
            marker="s",
            color=self.color_supporting,
            linewidth=2,
            label="Supporting attention",
            linestyle="-",
            alpha=0.75,
        )[0]
        ax_attn.fill_between(
            ns,
            np.asarray(supp_mean) - np.asarray(supp_sem),
            np.asarray(supp_mean) + np.asarray(supp_sem),
            color=self.color_supporting,
            alpha=0.15,
        )
        line_a = ax_acc.plot(
            ns,
            acc,
            marker="^",
            color=self.color_correct,
            linewidth=2,
            linestyle="--",
            label="Accuracy",
        )[0]
        ax_acc.fill_between(ns, acc_lo, acc_hi, color=self.color_correct, alpha=0.12)

        if show_values:
            for n, p in zip(ns, acc):
                if np.isfinite(p):
                    ax_acc.text(
                        n,
                        p,
                        f"{p:.2f}",
                        fontsize=8,
                        color="#222222",
                        ha="left",
                        va="bottom",
                    )

        ax_attn.set_xlabel("# distractor sentences", fontsize=11)
        ax_attn.set_ylabel("Mean attention (± SEM)", fontsize=11)
        ax_attn.set_xticks(ns)
        ax_attn.set_ylim(*attn_ylim(dist_mean, supp_mean))
        ax_attn.grid(axis="y", linestyle="--", alpha=0.35)
        ax_attn.set_axisbelow(True)

        ax_acc.set_ylabel("Accuracy (± 95% CI)", fontsize=11, color=self.color_correct)
        ax_acc.set_ylim(0, 1.02)
        ax_acc.tick_params(axis="y", colors=self.color_correct)
        ax_acc.spines["right"].set_color(self.color_correct)

        ax_attn.set_title(
            self._format_short_title(
                "Attention & accuracy vs distractor count", plot_name_add
            ),
            fontsize=11,
            pad=8,
        )
        ax_attn.legend(
            handles=[line_d, line_s, line_a],
            loc="upper left",
            fontsize=9,
            framealpha=0.9,
        )
        fig.tight_layout()

        png_path = self._resolve_save_target(
            f"distraction_vs_n_distractors_{version}.png", path_add
        )
        txt_rows = []
        for n, dm, ds, sm, ss, p, lo, hi, bn in zip(
            ns,
            dist_mean,
            dist_sem,
            supp_mean,
            supp_sem,
            acc,
            acc_lo,
            acc_hi,
            bin_n,
        ):
            txt_rows.append((f"n_dist={n} dist mean", dm))
            txt_rows.append((f"n_dist={n} dist sem", ds))
            txt_rows.append((f"n_dist={n} supp mean", sm))
            txt_rows.append((f"n_dist={n} supp sem", ss))
            txt_rows.append((f"n_dist={n} accuracy", p))
            txt_rows.append((f"n_dist={n} acc lo", lo))
            txt_rows.append((f"n_dist={n} acc hi", hi))
            txt_rows.append((f"n_dist={n} bin n", bn))
        self._write_plot_data_txt(png_path, [("Bin statistics", txt_rows)])
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_diff_two_runs_per_task(
        self,
        df: pd.DataFrame,
        y_label: str = "Difference",
        file_name: str | None = None,
        plot_name_add: list[str] | None = None,
    ) -> None:
        """
        Plot per-task difference between two runs (reas - da) for a given metric.
        """
        plt.close()

        plt.figure(figsize=(max(7, len(df["task_id"].unique()) * 0.75), 4.5))

        df = df.sort_values("task_id")
        ax = sns.barplot(
            data=df,
            x="task_id",
            y="diff",
            color=self.cmap(0),
        )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Task")
        ax.set_ylabel(y_label)
        ax.set_title(f"Difference of {y_label}", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        path_add_resolved = Path(*plot_name_add) if plot_name_add else None
        png_path = self._resolve_save_target(
            file_name or "diff_two_runs_per_task.png",
            path_add_resolved,
        )
        self._save_plot(file_name=png_path)

    def plot_toxic_cot_transition_overview(
        self,
        merged_df: pd.DataFrame,
        file_name: str = "toxic_cot_transition_overview.png",
        plot_name_add: list[str] = None,
        path_add: str | Path = "",
    ) -> None:
        """
        Plot an overview of toxic COT transitions between before and after versions.

        Categories:
        - appeared:     before=False, after=True
        - disappeared:  before=True,  after=False
        - stayed:       before=True,  after=True
        - never:        before=False, after=False
        """
        toxic_cot_before = merged_df["toxic_cot_before"].astype(bool)
        toxic_cot_after = merged_df["toxic_cot_after"].astype(bool)

        merged_df["toxic_cot_appeared"] = (~toxic_cot_before) & toxic_cot_after
        merged_df["toxic_cot_disappeared"] = toxic_cot_before & (~toxic_cot_after)
        merged_df["toxic_cot_stayed"] = toxic_cot_before & toxic_cot_after
        merged_df["toxic_cot_never"] = (~toxic_cot_before) & (~toxic_cot_after)

        category_order = [
            "toxic_cot_never",
            "toxic_cot_disappeared",
            "toxic_cot_appeared",
            "toxic_cot_stayed",
        ]
        category_labels = {
            "toxic_cot_never": "Never",
            "toxic_cot_disappeared": "Disappeared",
            "toxic_cot_appeared": "Appeared",
            "toxic_cot_stayed": "Stayed",
        }
        category_colors = {
            "toxic_cot_never": "#D9D9D9",
            "toxic_cot_disappeared": "#4CAF50",
            "toxic_cot_appeared": "#D55E00",
            "toxic_cot_stayed": "#7B3294",
        }

        counts = {cat: int(merged_df[cat].sum()) for cat in category_order}
        total = sum(counts.values())
        ratios = {cat: val / total if total else 0.0 for cat, val in counts.items()}

        fig, ax = plt.subplots(figsize=(11, 2.8))

        left = 0.0
        for cat in category_order:
            width = ratios[cat]
            ax.barh(
                y=["Toxic COT Transition"],
                width=[width],
                left=left,
                color=category_colors[cat],
                label=category_labels[cat],
                height=0.6,
            )

            if width > 0.04:
                ax.text(
                    left + width / 2,
                    0,
                    f"{category_labels[cat]}\n{counts[cat]} ({width:.1%})",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
            left += width

        before_rate = toxic_cot_before.mean()
        after_rate = toxic_cot_after.mean()
        delta = after_rate - before_rate

        title = "Toxic COT Transition Overview"
        if plot_name_add:
            title += f" ({'; '.join(plot_name_add)})"
        ax.set_title(title)

        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        x_label = (
            f"Share of parts  |  Before: {before_rate:.1%}   After: {after_rate:.1%}   "
            f"Delta: {delta:+.1%}"
        )
        ax.set_xlabel(x_label)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Transition")
        plt.tight_layout()

        png_path = self._resolve_save_target(file_name, path_add if path_add else None)
        self._save_plot(file_name=png_path)

    def plot_attr_before_after_two_runs_per_task(
        self,
        vals_da: list[dict[int, float]],
        vals_reas: list[dict[int, float]],
        y_label: str = "Value",
        file_name: str | None = None,
        plot_name_add: list[str] | None = None,
    ) -> None:
        """
        Plot per-task grouped bars for a metric:
        - direct answer (before and after)
        - reasoning (before and after)
        'After' version can be omitted.
        """
        assert len(vals_reas) == len(
            vals_da
        ), "vals_reas and vals_da must have the same number of versions"
        # Ensure all dicts have the same task keys
        tasks = range(1, 21)

        x = np.arange(len(tasks))
        width = 0.2
        colors = self.cmap(np.linspace(0, 1, 4))

        da_before = [vals_da[0][t] for t in tasks]
        reas_before = [vals_reas[0][t] for t in tasks]
        reas_after, da_after = None, None
        if len(vals_reas) > 1:
            da_after = [vals_da[1][t] for t in tasks]
            reas_after = [vals_reas[1][t] for t in tasks]

        plt.figure(figsize=(14, 5))

        plt.bar(
            x - 1.5 * width,
            reas_before,
            width,
            label="Reasoning before",
            color=colors[0],
        )
        plt.bar(x + 0.5 * width, da_before, width, label="DA before", color=colors[2])
        if reas_after and da_after:
            plt.bar(x + 1.5 * width, da_after, width, label="DA after", color=colors[3])
            plt.bar(
                x - 0.5 * width,
                reas_after,
                width,
                label="Reasoning after",
                color=colors[1],
            )

        # Use your standard formatting helper
        self._plot_general_details(
            x_label="Task",
            y_label=y_label,
            max_x_len=len(tasks),
            plot_name_add=plot_name_add,
            num_of_data_arrays=4,
            step=1,
        )

        plt.xticks(x, tasks)
        plt.legend(loc="upper right")

        fn = file_name or f"{y_label.replace(' ', '_').lower()}_two_runs_per_task.png"
        path_add = Path("/".join(plot_name_add)) if plot_name_add else None
        png_path = self._resolve_save_target(fn, path_add)
        self._save_plot(file_name=png_path)

    def plot_attrs_by_runs_versions_toxicity(
        self,
        df: pd.DataFrame,
        attrs: list[str],
        multi_system: bool,
    ):
        # TODO: Redo this method (currently doesn't plot counts from separate runs properly)
        # plot_df = []
        #
        # for attr in attrs:
        #     cols = [
        #         "task_id",
        #         "toxic_cot_before",
        #         f"{attr}_before_da",
        #         f"{attr}_before_reas",
        #     ]
        #     if multi_system:
        #         cols += ["toxic_cot_after", f"{attr}_after_da", f"{attr}_after_reas"]
        #
        #     tmp = df[cols].copy()
        #     tmp["attr"] = attr
        #
        #     tmp["da_toxic_before"] = tmp["toxic_cot_before"]
        #     tmp["da_not_toxic_before"] = ~tmp["toxic_cot_before"]
        #     tmp["reas_toxic_before"] = tmp["toxic_cot_before"]
        #     tmp["reas_not_toxic_before"] = ~tmp["toxic_cot_before"]
        #
        #     if multi_system:
        #         tmp["da_toxic_after"] = tmp["toxic_cot_after"]
        #         tmp["da_not_toxic_after"] = ~tmp["toxic_cot_after"]
        #         tmp["reas_toxic_after"] = tmp["toxic_cot_after"]
        #         tmp["reas_not_toxic_after"] = ~tmp["toxic_cot_after"]
        #
        #     plot_df.append(tmp)
        #
        # plot_df = pd.concat(plot_df, ignore_index=True)
        rows = []

        for attr in attrs:
            for run in ["da", "reas"]:
                before_col = f"{attr}_before_{run}"
                after_col = f"{attr}_after_{run}" if multi_system else None

                cols = ["task_id", f"toxic_cot_before", before_col]
                if multi_system:
                    cols += [f"toxic_cot_after", after_col]

                tmp = df[cols].copy()
                tmp["attr"] = attr
                tmp["run_type"] = run

                tmp = tmp.rename(columns={before_col: "attr_value_before"})
                if multi_system:
                    tmp = tmp.rename(columns={after_col: "attr_value_after"})

                rows.append(tmp)

        plot_df = pd.concat(rows, ignore_index=True)
        # value_cols = [c for c in plot_df.columns if c.startswith(("da_", "reas_"))]
        # long_df = plot_df.melt(
        #     id_vars=["task_id", "attr"],
        #     value_vars=value_cols,
        #     var_name="group",
        #     value_name="flag",
        # )
        # summary = (
        #     long_df.groupby(["task_id", "attr", "group"])["flag"]
        #     .mean()
        #     .reset_index(name="percentage")
        # )
        before_summary = (
            plot_df.groupby(["attr", "run_type"])["toxic_cot_before"]
            .agg(
                toxic_count="sum",
                total_count="size",
            )
            .reset_index()
        )
        before_summary["non_toxic_count"] = (
            before_summary["total_count"] - before_summary["toxic_count"]
        )
        # For percentages
        # before_summary = (
        #     plot_df.groupby(["task_id", "attr", "run_type"])["toxic_cot_before"]
        #     .mean()
        #     .reset_index(name="toxic_rate")
        # )
        axes = None
        after_summary = None
        if multi_system:
            after_summary = (
                plot_df.groupby(["attr", "run_type"])["toxic_cot_after"]
                .agg(
                    toxic_count="sum",
                    total_count="size",
                )
                .reset_index()
            )
            after_summary["non_toxic_count"] = (
                after_summary["total_count"] - after_summary["toxic_count"]
            )
            fig, axes = plt.subplots(2, 1, figsize=(10, 5))

        plot_df = before_summary.melt(
            id_vars=["attr", "run_type"],
            value_vars=["toxic_count", "non_toxic_count"],
            var_name="toxicity",
            value_name="count",
        )
        g = sns.catplot(
            data=plot_df,
            x="attr",
            y="count",
            hue="run_type",
            col="toxicity",
            kind="bar",
            height=4,
            aspect=1.4,
        )

        g.set_axis_labels("Attribute", "Count")
        # g.legend.set_title("Toxic CoT")
        g.set_xticklabels(rotation=45, ha="right")

        if multi_system:
            g = sns.catplot(
                data=after_summary,
                x="run_type",
                y="count",
                hue="toxic_cot_before",
                # col="task_id",
                row="attr",
                kind="bar",
                height=4,
                aspect=1.4,
                ax=axes[1],
            )
        self._save_plot(file_name="attrs_by_runs_versions_toxicity.png")

    def plot_accuracy_vs_distraction_ratio(
        self,
        stats: DistractorAttentionStats,
        plot_name_add: list[str] | None = None,
        version: str = "before",
        n_bins: int = 6,
        min_bin_size: int = 3,
        eps: float = 1e-8,
        path_add: Path | None = None,
        show_values: bool = False,
    ) -> None:
        """
        Accuracy as a function of the per-sample distractor / supporting
        attention ratio. Bins are evenly spaced in log-ratio. Error bars are
        95% normal-approximation CIs.

        :param stats: accumulated distractor attention records for this version
        :param plot_name_add: extra tags appended to the title (in brackets)
        :param version: "before" or "after"; appears in the filename
        :param n_bins: number of log-ratio bins
        :param min_bin_size: minimum number of records per bin to plot it
        :param eps: stabiliser for division
        :param path_add: sub-folder under results_path
        :param show_values: when True, label each point with its accuracy
        """
        ratios: list[float] = []
        correct: list[bool] = []
        for r in stats.records:
            if r.attn_distractor is None or r.attn_supporting is None:
                continue
            ratio = r.attn_distractor / (r.attn_supporting + eps)
            if ratio <= 0:
                continue
            ratios.append(ratio)
            correct.append(bool(r.answer_correct))

        if not ratios:
            print(
                f"[plot_accuracy_vs_distraction_ratio] No data for version='{version}'."
            )
            return

        ratios_arr = np.asarray(ratios)
        correct_arr = np.asarray(correct)

        # Use the same fixed log-range as the ratio boxplot so the plots line up.
        lo_log = np.log10(RATIO_YMIN)
        hi_log = np.log10(RATIO_YMAX)
        edges = np.linspace(lo_log, hi_log, n_bins + 1)

        log_r = np.clip(np.log10(ratios_arr), edges[0], edges[-1] - 1e-9)
        bin_idx = np.clip(np.digitize(log_r, edges) - 1, 0, n_bins - 1)

        def _normal_ci(k: int, n: int, z: float = 1.96):
            if n == 0:
                return float("nan"), float("nan"), float("nan")
            p = k / n
            se = np.sqrt(p * (1 - p) / n)
            return p, max(0.0, p - z * se), min(1.0, p + z * se)

        bin_centres, bin_acc, bin_lo, bin_hi, bin_n = [], [], [], [], []
        for b in range(n_bins):
            mask = bin_idx == b
            n_b = int(mask.sum())
            if n_b < min_bin_size:
                continue
            k_b = int(correct_arr[mask].sum())
            p, lo_p, hi_p = _normal_ci(k_b, n_b)
            centre = 10 ** ((edges[b] + edges[b + 1]) / 2)
            bin_centres.append(centre)
            bin_acc.append(p)
            bin_lo.append(lo_p)
            bin_hi.append(hi_p)
            bin_n.append(n_b)

        if not bin_centres:
            print(
                f"[plot_accuracy_vs_distraction_ratio] No bins meet "
                f"min_bin_size={min_bin_size} for version='{version}'."
            )
            return

        fig, ax = plt.subplots(figsize=(7.5, 5))
        bin_acc = np.asarray(bin_acc)
        bin_lo = np.asarray(bin_lo)
        bin_hi = np.asarray(bin_hi)
        yerr = np.vstack([bin_acc - bin_lo, bin_hi - bin_acc])

        ax.errorbar(
            bin_centres,
            bin_acc,
            yerr=yerr,
            fmt="o-",
            color="#2c3e50",
            ecolor="#7f8c8d",
            capsize=4,
            linewidth=1.8,
            markersize=7,
        )

        # Shade left/right of ratio = 1.
        ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.axvspan(RATIO_YMIN, 1.0, facecolor=self.color_correct, alpha=0.05, zorder=0)
        ax.axvspan(
            1.0,
            RATIO_YMAX,
            facecolor=self.color_incorrect,
            alpha=0.05,
            zorder=0,
        )

        if show_values:
            for x, p in zip(bin_centres, bin_acc):
                if np.isfinite(p):
                    ax.text(
                        x,
                        p + 0.02,
                        f"{p:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="#222222",
                    )

        ax.set_xscale("log")
        ax.set_xlim(RATIO_YMIN, RATIO_YMAX)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(
            "Ratio of attention paid to distractor sentences vs. supporting sentences (log)",
            fontsize=11,
        )
        ax.set_ylabel("Accuracy (± 95% CI)", fontsize=11)
        ax.set_title(
            self._format_short_title("Accuracy vs distractor ratio", plot_name_add),
            fontsize=11,
            pad=8,
        )
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()

        png_path = self._resolve_save_target(
            f"accuracy_vs_distraction_ratio_{version}.png", path_add
        )
        txt_rows = []
        for c, p, lo, hi, n in zip(bin_centres, bin_acc, bin_lo, bin_hi, bin_n):
            txt_rows.append((f"ratio_centre={c:.4f} accuracy", float(p)))
            txt_rows.append((f"ratio_centre={c:.4f} acc_lo", float(lo)))
            txt_rows.append((f"ratio_centre={c:.4f} acc_hi", float(hi)))
            txt_rows.append((f"ratio_centre={c:.4f} n", int(n)))
        self._write_plot_data_txt(png_path, [("Bin statistics", txt_rows)])
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def _disambiguator_from_tags(self, plot_name_add: list[str] | None) -> str:
        """
        Build a filename-safe disambiguator suffix from ``plot_name_add``.

        Drops the ``"before"`` / ``"after"`` version tags (which already appear
        in the filename) and joins the remaining tags with underscores, prefixed
        by ``"_"``.  Returns ``""`` when there is nothing meaningful to add.

        :param plot_name_add: optional list of context tags from the caller
        :return: e.g. ``"_Split-valid"`` or ``""``
        """
        if not plot_name_add:
            return ""
        tags = [t for t in plot_name_add if t.lower() not in ("before", "after")]
        if not tags:
            return ""
        safe = "_".join(
            t.replace(" ", "_").replace("/", "-").replace("\\", "-") for t in tags
        )
        return f"_{safe}"

    def _ba_pick_evaluators(
        self,
        evaluators: list,
        versions: list[str] | None = None,
    ) -> tuple:
        """
        Pick the *before* and *after* evaluators from a list.

        When *versions* is provided (the preferred path) it is used directly:
        each evaluator is paired with its version string by position, and the
        first one whose version contains ``"before"`` / ``"after"`` is chosen.

        :param evaluators: list of MetricEvaluator objects (parallel to versions)
        :param versions: optional list of version strings, e.g. ``["before", "after"]``
        :return: (before, after) — either may be ``None``
        """
        before, after = None, None

        if versions is not None:
            for ev, v in zip(evaluators, versions):
                v_low = v.lower()
                if "before" in v_low:
                    before = ev
                elif "after" in v_low:
                    after = ev

        if before is None and after is None:
            # Positional fallback: single-system → after only; two → first/second
            if len(evaluators) == 1:
                after = evaluators[0]
            elif len(evaluators) >= 2:
                before, after = evaluators[0], evaluators[1]
        elif after is None:
            # We found a "before" but no "after" — use the last remaining one
            remaining = [ev for ev in evaluators if ev is not before]
            after = remaining[-1] if remaining else None

        return before, after

    def _ba_per_task(
        self,
        evaluator,
        mean_attr: str,
        std_attr: str | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Extract per-task mean and standard-deviation arrays from an evaluator.

        Looks for *mean_attr* (and optionally *std_attr*) on the evaluator.
        The attribute is expected to be a ``Metric``-like object with an
        ``.all`` list of per-task values, or a plain list/array.

        :param evaluator: a MetricEvaluator, or ``None``
        :param mean_attr: attribute name for the mean metric
        :param std_attr: attribute name for the std metric, or ``None``
        :return: (means_array, stds_array_or_None)
        """
        if evaluator is None:
            return np.array([]), None

        metric = getattr(evaluator, mean_attr, None)
        if metric is None:
            return np.array([]), None

        try:
            raw = metric.all if hasattr(metric, "all") else list(metric)
            means = np.array([float(v) if v is not None else np.nan for v in raw])
        except Exception:
            return np.array([]), None

        stds = None
        if std_attr:
            std_metric = getattr(evaluator, std_attr, None)
            if std_metric is not None:
                try:
                    raw_std = (
                        std_metric.all
                        if hasattr(std_metric, "all")
                        else list(std_metric)
                    )
                    stds = np.array(
                        [float(v) if v is not None else np.nan for v in raw_std]
                    )
                except Exception:
                    stds = None

        return means, stds

    def _ba_plot_lines(
        self,
        ax,
        before,
        after,
        mean_attr: str,
        std_attr: str | None,
        ylabel: str,
        ylim: tuple[float, float] | None,
        show_values: bool = False,
    ) -> list[tuple[str, float | None]]:
        """
        Draw before/after lines with optional std bands onto *ax*.

        Each version is drawn with a distinct colour (blue for *before*, orange
        for *after*).  Missing evaluators are silently skipped.

        :param ax: matplotlib ``Axes`` to draw on
        :param before: *before* MetricEvaluator, or ``None``
        :param after: *after* MetricEvaluator, or ``None``
        :param mean_attr: evaluator attribute name for the per-task mean values
        :param std_attr: evaluator attribute name for the per-task std values,
                         or ``None`` to skip the band
        :param ylabel: y-axis label (empty string → no label set)
        :param ylim: ``(ymin, ymax)`` passed to ``ax.set_ylim``; ``None`` →
                     data-driven limits
        :param show_values: when ``True``, annotate each point with its value
        :return: list of ``(label, value)`` pairs for the companion ``.txt``
                 file
        """
        rows: list[tuple[str, float | None]] = []
        specs = [
            (before, self.color_supporting, "Before"),
            (after, self.color_distractor, "After"),
        ]

        max_n = 0
        for evaluator, color, label in specs:
            means, stds = self._ba_per_task(evaluator, mean_attr, std_attr)
            if len(means) == 0:
                continue
            max_n = max(max_n, len(means))
            x = np.arange(1, len(means) + 1)
            ax.plot(
                x,
                means,
                marker="o",
                color=color,
                linewidth=2,
                label=label,
                zorder=3,
            )
            if stds is not None and len(stds) == len(means):
                ax.fill_between(
                    x,
                    means - stds,
                    means + stds,
                    color=color,
                    alpha=0.15,
                    zorder=2,
                )
            for xi, v in zip(x, means):
                val = float(v) if np.isfinite(v) else None
                rows.append((f"task={xi} {label}", val))
                if show_values and val is not None:
                    ax.text(
                        xi,
                        v,
                        f"{v:.2f}",
                        fontsize=7,
                        ha="center",
                        va="bottom",
                        color="#222222",
                    )

        ax.set_xlabel("Task", fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)
        if max_n > 0:
            ax.set_xticks(np.arange(1, max_n + 1))
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        return rows

    def plot_before_after_accuracy(
        self,
        evaluators: list,
        versions: list[str] | None = None,
        plot_name_add: list[str] | None = None,
        path_add: Path | None = None,
        show_values: bool = False,
    ) -> None:
        """
        Compare exact-match and soft-match accuracy before vs after, per task.

        Two side-by-side panels: exact-match on the left, soft-match on the
        right. Each panel shows one line per version with std bands.

        :param evaluators: ``split.evaluators`` (one MetricEvaluator per version)
        :param versions: ``split.versions`` — version strings parallel to evaluators,
                         e.g. ``["before", "after"]``; used to identify which
                         evaluator is which. If omitted falls back to position.
        :param plot_name_add: extra tags appended to the title
        :param path_add: sub-folder under results_path
        :param show_values: when True, label each point with its value
        """
        before, after = self._ba_pick_evaluators(evaluators, versions)
        if before is None and after is None:
            print("[plot_before_after_accuracy] No evaluators provided.")
            return

        fig, (ax_em, ax_sm) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

        em_rows = self._ba_plot_lines(
            ax_em,
            before,
            after,
            mean_attr="exact_match_accuracy",
            std_attr="exact_match_std",
            ylabel="Accuracy",
            ylim=(0.0, 1.05),
            show_values=show_values,
        )
        ax_em.set_title("Exact match", fontsize=11)
        ax_em.legend(fontsize=9, loc="lower right", framealpha=0.9)

        sm_rows = self._ba_plot_lines(
            ax_sm,
            before,
            after,
            mean_attr="soft_match_accuracy",
            std_attr="soft_match_std",
            ylabel="",
            ylim=(0.0, 1.05),
            show_values=show_values,
        )
        ax_sm.set_title("Soft match", fontsize=11)

        fig.suptitle(
            self._format_short_title("Accuracy per version", plot_name_add),
            fontsize=12,
        )
        fig.tight_layout()

        png_path = self._resolve_save_target(
            f"split_accuracy{self._disambiguator_from_tags(plot_name_add)}.png",
            path_add,
        )
        self._write_plot_data_txt(
            png_path,
            [("Exact match", em_rows), ("Soft match", sm_rows)],
        )
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_before_after_reasoning_scores(
        self,
        evaluators: list,
        versions: list[str] | None = None,
        plot_name_add: list[str] | None = None,
        path_add: Path | None = None,
        show_values: bool = False,
    ) -> None:
        """
        Compare BLEU, ROUGE, and METEOR before vs after, per task.

        Three side-by-side panels with shared y-axis (all three live in [0,1]),
        one line per version each.

        :param evaluators: ``split.evaluators`` (one MetricEvaluator per version)
        :param versions: ``split.versions`` — version strings parallel to evaluators
        :param plot_name_add: extra tags appended to the title
        :param path_add: sub-folder under results_path
        :param show_values: when True, label each point with its value
        """
        before, after = self._ba_pick_evaluators(evaluators, versions)
        if before is None and after is None:
            print("[plot_before_after_reasoning_scores] No evaluators provided.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
        sections: list[tuple[str, list[tuple[str, float | None]]]] = []

        score_specs = [
            ("BLEU", "bleu", "bleu_std"),
            ("ROUGE", "rouge", "rouge_std"),
            ("METEOR", "meteor", "meteor_std"),
        ]
        for ax, (title, mean_attr, std_attr) in zip(axes, score_specs):
            rows = self._ba_plot_lines(
                ax,
                before,
                after,
                mean_attr=mean_attr,
                std_attr=std_attr,
                ylabel="Score" if ax is axes[0] else "",
                ylim=(0.0, 1.05),
                show_values=show_values,
            )
            ax.set_title(title, fontsize=11)
            sections.append((title, rows))

        axes[0].legend(fontsize=9, loc="lower right", framealpha=0.9)
        fig.suptitle(
            self._format_short_title("Reasoning scores per version", plot_name_add),
            fontsize=12,
        )
        fig.tight_layout()

        png_path = self._resolve_save_target(
            f"split_reasoning_scores{self._disambiguator_from_tags(plot_name_add)}.png",
            path_add,
        )
        self._write_plot_data_txt(png_path, sections)
        self._save_plot(file_name=png_path)
        plt.close(fig)

        # --- Second view: one panel per version, all three scores as lines ---
        # Complements the per-score view by making it easy to compare BLEU/ROUGE/
        # METEOR within a single version at a glance (mirroring how attention plots
        # overlay max_supp_attn and attn_on_target on one axis).
        version_evs = [
            (lbl, ev)
            for lbl, ev in [("Before", before), ("After", after)]
            if ev is not None
        ]
        if version_evs:
            n_ver = len(version_evs)
            fig2, axes2 = plt.subplots(
                1, n_ver, figsize=(7 * n_ver, 4.5), sharey=True, squeeze=False
            )
            score_line_specs = [
                ("BLEU", "bleu", "bleu_std", self.color_supporting, "o"),
                ("ROUGE", "rouge", "rouge_std", self.color_distractor, "s"),
                ("METEOR", "meteor", "meteor_std", self.color_neutral, "^"),
            ]
            sections2: list[tuple[str, list[tuple[str, float | None]]]] = []
            for ax2, (vlabel, ev) in zip(axes2[0], version_evs):
                rows2: list[tuple[str, float | None]] = []
                max_n2 = 0
                for slabel, mean_attr, std_attr, color, marker in score_line_specs:
                    means2, stds2 = self._ba_per_task(ev, mean_attr, std_attr)
                    if len(means2) == 0:
                        continue
                    max_n2 = max(max_n2, len(means2))
                    x2 = np.arange(1, len(means2) + 1)
                    ax2.plot(
                        x2,
                        means2,
                        marker=marker,
                        color=color,
                        linewidth=2,
                        label=slabel,
                    )
                    if stds2 is not None and len(stds2) == len(means2):
                        ax2.fill_between(
                            x2, means2 - stds2, means2 + stds2, color=color, alpha=0.15
                        )
                    for xi, v in zip(x2, means2):
                        rows2.append(
                            (
                                f"task={xi} {slabel}",
                                float(v) if np.isfinite(v) else None,
                            )
                        )
                ax2.set_title(vlabel, fontsize=11)
                ax2.set_xlabel("Task", fontsize=10)
                ax2.set_ylabel("Score" if ax2 is axes2[0][0] else "", fontsize=10)
                ax2.set_ylim(0, 1.05)
                if max_n2 > 0:
                    ax2.set_xticks(np.arange(1, max_n2 + 1))
                ax2.legend(fontsize=9, loc="lower right", framealpha=0.9)
                ax2.grid(axis="y", linestyle="--", alpha=0.4)
                ax2.set_axisbelow(True)
                sections2.append((vlabel, rows2))

            fig2.suptitle(
                self._format_short_title("Reasoning scores by version", plot_name_add),
                fontsize=12,
            )
            fig2.tight_layout()
            png_path2 = self._resolve_save_target(
                f"split_reasoning_scores_by_version"
                f"{self._disambiguator_from_tags(plot_name_add)}.png",
                path_add,
            )
            self._write_plot_data_txt(png_path2, sections2)
            self._save_plot(file_name=png_path2)
            plt.close(fig2)

    def plot_before_after_attention(
        self,
        evaluators: list,
        versions: list[str] | None = None,
        plot_name_add: list[str] | None = None,
        path_add: Path | None = None,
        show_values: bool = False,
    ) -> None:
        """
        Compare max-supporting attention and attention-on-target before vs
        after, per task.

        :param evaluators: ``split.evaluators`` (one MetricEvaluator per version)
        :param versions: ``split.versions`` — version strings parallel to evaluators
        :param plot_name_add: extra tags appended to the title
        :param path_add: sub-folder under results_path
        :param show_values: when True, label each point with its value
        """
        before, after = self._ba_pick_evaluators(evaluators, versions)
        if before is None and after is None:
            print("[plot_before_after_attention] No evaluators provided.")
            return

        fig, (ax_max, ax_target) = plt.subplots(1, 2, figsize=(13, 4.5))

        max_rows = self._ba_plot_lines(
            ax_max,
            before,
            after,
            mean_attr="max_supp_attn",
            std_attr="max_supp_attn_std",
            ylabel="Mean attention",
            ylim=None,  # data-driven; _ba_plot_lines handles None
            show_values=show_values,
        )
        ax_max.set_title("Max attention on supporting", fontsize=11)
        ax_max.legend(fontsize=9, loc="upper right", framealpha=0.9)

        target_rows = self._ba_plot_lines(
            ax_target,
            before,
            after,
            mean_attr="attn_on_target",
            std_attr="attn_on_target_std",
            ylabel="Attention",
            ylim=None,  # this metric is unbounded
            show_values=show_values,
        )
        ax_target.set_title("Attention on target tokens", fontsize=11)

        fig.suptitle(
            self._format_short_title("Attention per version", plot_name_add),
            fontsize=12,
        )
        fig.tight_layout()

        png_path = self._resolve_save_target(
            f"split_attention{self._disambiguator_from_tags(plot_name_add)}.png",
            path_add,
        )
        self._write_plot_data_txt(
            png_path,
            [
                ("Max attention on supporting", max_rows),
                ("Attention on target", target_rows),
            ],
        )
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_before_after_summary(
        self,
        evaluators: list,
        versions: list[str] | None = None,
        plot_name_add: list[str] | None = None,
        path_add: Path | None = None,
        show_values: bool = False,
    ) -> None:
        """
        Compact dashboard summarising all metrics: one grouped bar chart with
        the mean (across tasks) of each metric, before vs after side by side.

        Useful as a single at-a-glance comparison; the per-task detail lives
        in the other three before-after plots.

        :param evaluators: ``split.evaluators`` (one MetricEvaluator per version)
        :param versions: ``split.versions`` — version strings parallel to evaluators
        :param plot_name_add: extra tags appended to the title
        :param path_add: sub-folder under results_path
        :param show_values: when True, label each bar with its value
        """
        before, after = self._ba_pick_evaluators(evaluators, versions)
        if before is None and after is None:
            print("[plot_before_after_summary] No evaluators provided.")
            return

        # (display_label, attribute_name, group_name)
        spec: list[tuple[str, str, str]] = [
            ("Exact match", "exact_match_accuracy", "Accuracy"),
            ("Soft match", "soft_match_accuracy", "Accuracy"),
            ("BLEU", "bleu", "Reasoning"),
            ("ROUGE", "rouge", "Reasoning"),
            ("METEOR", "meteor", "Reasoning"),
            ("Max-supp attn", "max_supp_attn", "Attention"),
            ("Attn-on-target", "attn_on_target", "Attention"),
        ]

        labels = [s[0] for s in spec]
        n = len(labels)
        x = np.arange(n)
        width = 0.36

        def _means(evaluator) -> list[float | None]:
            if evaluator is None:
                return [None] * n
            out: list[float | None] = []
            for _, attr, _ in spec:
                metric = getattr(evaluator, attr, None)
                if metric is None:
                    out.append(None)
                    continue
                try:
                    out.append(float(metric.get_mean()))
                except Exception:
                    out.append(None)
            return out

        before_means = _means(before)
        after_means = _means(after)

        fig, ax = plt.subplots(figsize=(11, 4.8))

        def _plot_bars(offset, values, color, label):
            xs, hs = [], []
            for xi, v in zip(x, values):
                if v is None or not np.isfinite(v):
                    continue
                xs.append(xi + offset)
                hs.append(v)
            if not xs:
                return None
            bars = ax.bar(xs, hs, width, color=color, alpha=0.82, label=label)
            if show_values:
                for b, h in zip(bars, hs):
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        h,
                        f"{h:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#222222",
                    )
            return bars

        _plot_bars(-width / 2, before_means, self.color_supporting, "Before")
        _plot_bars(+width / 2, after_means, self.color_distractor, "After")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Mean across tasks", fontsize=11)
        # Most metrics live in [0, 1] but attn_on_target can exceed 1.
        # Pick the y-limit dynamically so no bar gets clipped, but keep the
        # baseline at 1.05 so different runs are still roughly comparable.
        all_vals = [
            v for v in (before_means + after_means) if v is not None and np.isfinite(v)
        ]
        ymax = max(1.05, max(all_vals) * 1.10) if all_vals else 1.05
        ax.set_ylim(0, ymax)
        ax.set_title(
            self._format_short_title("Summary", plot_name_add),
            fontsize=12,
            pad=8,
        )
        ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()

        png_path = self._resolve_save_target(
            f"split_summary{self._disambiguator_from_tags(plot_name_add)}.png",
            path_add,
        )
        rows: list[tuple[str, float | None]] = []
        for label, b, a in zip(labels, before_means, after_means):
            rows.append((f"{label} before", b))
            rows.append((f"{label} after", a))
            if b is not None and a is not None:
                rows.append((f"{label} delta", float(a) - float(b)))
        self._write_plot_data_txt(png_path, [("Mean across tasks", rows)])
        self._save_plot(file_name=png_path)
        plt.close(fig)

    def plot_before_after_delta_lineplot(
        self,
        evaluators: list,
        versions: list[str] | None = None,
        plot_name_add: list[str] | None = None,
        path_add: Path | None = None,
        show_values: bool = False,
    ) -> None:
        """
        Plots the absolute delta (after - before) for exact match and soft match accuracy
        per task as a line plot. Positive delta means 'after' is better.

        :param evaluators: ``split.evaluators`` (one MetricEvaluator per version)
        :param versions: ``split.versions`` — version strings parallel to evaluators
        """
        before, after = self._ba_pick_evaluators(evaluators, versions)
        if before is None or after is None:
            print(
                "[plot_before_after_delta_lineplot] Both 'before' and 'after' evaluators are required."
            )
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        em_before, _ = self._ba_per_task(before, "exact_match_accuracy", None)
        em_after, _ = self._ba_per_task(after, "exact_match_accuracy", None)
        sm_before, _ = self._ba_per_task(before, "soft_match_accuracy", None)
        sm_after, _ = self._ba_per_task(after, "soft_match_accuracy", None)

        x = np.arange(1, len(em_before) + 1)
        if len(em_before) > 0 and len(em_after) == len(em_before):
            delta_em = em_after - em_before
            ax.plot(
                x,
                delta_em,
                marker="o",
                color="#2874A6",
                linewidth=2,
                label="Exact Match Delta",
            )
            if show_values:
                for xi, yi in zip(x, delta_em):
                    ax.text(xi, yi, f"{yi:+.2f}", fontsize=8, ha="center", va="bottom")

        if len(sm_before) > 0 and len(sm_after) == len(sm_before):
            delta_sm = sm_after - sm_before
            ax.plot(
                x,
                delta_sm,
                marker="s",
                color="#E67E22",
                linewidth=2,
                label="Soft Match Delta",
            )
            if show_values:
                for xi, yi in zip(x, delta_sm):
                    ax.text(xi, yi, f"{yi:+.2f}", fontsize=8, ha="center", va="top")

        ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Task", fontsize=11)
        ax.set_ylabel("Accuracy Delta (After - Before)", fontsize=11)
        ax.set_title(
            self._format_short_title("Accuracy Delta per Task", plot_name_add),
            fontsize=12,
            pad=8,
        )
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="both", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        fig.tight_layout()
        png_path = self._resolve_save_target(
            f"before_after_delta_lineplot{self._disambiguator_from_tags(plot_name_add)}.png",
            path_add,
        )
        self._save_plot(file_name=png_path)
        plt.close(fig)
