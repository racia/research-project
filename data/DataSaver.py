from __future__ import annotations

import csv
import sys
from typing import TextIO, Union, Iterable

from data.utils import *
from evaluation.Evaluator import MetricEvaluator
from inference.DataLevels import Task, Features, Split
from inference.Prompt import Prompt
from settings.config import DataSplits


class DataSaver:
    """
    This class handles everything related to saving data.
    """

    def __init__(self, save_to: str) -> None:
        """
        Initialize the DataSaver.
        The datasaver handles everything related to saving data.

        :param save_to: the path to save the data
        """
        self.old_stdout: TextIO = sys.stdout
        # self.results_path is updated in create_result_paths
        self.results_path: Path = Path(save_to)
        self.run_path: Path = Path(save_to)
        self.prompt_name: str = ""

    def create_result_paths(
        self,
        prompt_name: str,
        splits: list[Union[DataSplits.train, DataSplits.valid, DataSplits.test]],
    ) -> tuple[str, dict[str, str], dict[str, str]]:
        """
        Create the unique results path for the run and the files to save the data:
        log file, results file, and metrics files. The results path is also updated to match the current prompt.

        :param prompt_name: the name of the prompt
        :param splits: splits of the data
        :return: the names for the log file, result files, and metric files in a tuple
        """
        self.results_path = self.run_path / prompt_name

        try:
            os.makedirs(self.results_path)
        except FileExistsError:
            print(
                f"Directory {self.results_path} already exists and is not empty. "
                "Please choose another results_path or empty the directory."
            )
            self.results_path = Path("./outputs") / prompt_name
            os.makedirs(self.results_path)
        except OSError:
            print(
                f"Creation of the directory {self.results_path} failed due "
                f"to lack of writing rights. Please check the path."
            )
            self.results_path = Path("./outputs") / prompt_name
            os.makedirs(self.results_path)

        print(f"\nThe results will be saved to {self.results_path}\n")

        results_file_names = {}
        metrics_file_names = {}

        for split in splits:
            results_file_names[split] = f"{split}_{prompt_name}_results.csv"
            metrics_file_names[split] = f"{split}_{prompt_name}_metrics.csv"

        log_file_name = f"{prompt_name}.log"

        return log_file_name, results_file_names, metrics_file_names

    def save_output(
        self,
        data: list[dict[str, str | int | float]],
        headers: list | tuple,
        file_name: str | Path,
        path_add: str = "",
        flag: str = "a+",
    ) -> None:
        """
        This function allows to save the data continuously throughout the run.
        The headers are added once at the beginning, and the data is appended
        to the end of the file.

        This is how DictWriter works:
            headers = ['first_name', 'last_name']
            row = {'first_name': 'Lovely', 'last_name': 'Spam'}
            writer.writerow(row)

        :param data: one row as list of strings or multiple such rows
        :param headers: the headers for the csv file
        :param file_name: the name of the file to save the data
        :param path_add: an addition to the results path (goes between results_path and file_name)
        :param flag: the flag to open the file
        :return: None
        """
        if isinstance(file_name, str):
            file_name = self.results_path / path_add / file_name
            Path(self.results_path / path_add).mkdir(parents=True, exist_ok=True)
            if file_name.suffix != ".csv":
                raise ValueError("The file should be saved in a .csv format.")
        else:
            file_name = Path(file_name)

        with open(file_name, flag, encoding="UTF-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t")
            if is_empty_file(file_name):
                writer.writeheader()
            [writer.writerow(row) for row in data]

    def save_split_accuracy(
        self,
        evaluator: MetricEvaluator,
        metrics_file_name: str,
        after: bool = True,
    ) -> None:
        """
        Save the accuracies for the split,
        including the mean accuracy for all tasks.

        :param evaluator: the evaluator
        :param metrics_file_name: the name of the file to save the accuracies
        :param after: if to save the accuracy for after the setting was applied
        :return: None
        """
        accuracies_to_save = list(
            format_accuracy_metrics(
                evaluator.exact_match_accuracy,
                evaluator.soft_match_accuracy,
                evaluator.exact_match_std,
                evaluator.soft_match_std,
                after=after,
            ).values()
        )
        headers = list(accuracies_to_save[0].keys())
        self.save_output(
            data=accuracies_to_save,
            headers=headers,
            file_name=metrics_file_name,
        )

    def save_split_metrics(
        self, features: Features, metrics_file_names: list[str]
    ) -> None:
        """
        Save the metrics for all the tasks in a split.

        :param features: the features to save
        :param metrics_file_names: the path to save the metrics
        :return: None
        """
        headers = ["id", "task_id"]
        features = [
            {h: m for h, m in zip(headers, metric)} for metric in features.get().items()
        ]
        for result_file_name in metrics_file_names:
            self.save_output(
                data=features,
                headers=headers,
                file_name=result_file_name,
                # TODO: check if leaving it out works for multi_system
                # path_add="after" if after else "before",
            )

    @staticmethod
    def save_with_separator(file_path: Path, data: Iterable, sep="\n") -> None:
        """
        Save the separator between the data.

        :param file_path: the path to the file
        :param data: the data to save
        :param sep: the separator, a newline by default
        :return: None
        """
        with open(file_path, "w", encoding="UTF-8") as file:
            file.write(sep.join(map(lambda x: str(x).strip(), data)))

    def save_interpretability(self, task_data: Task, after: bool = True) -> None:
        """
        Save the interpretability result per sample part.

        :param task_data: the task instance with the results
        :param after: if to save the interpretability result for after the setting was applied
        :return: None
        """
        attn_scores_subdir = (
            self.results_path
            / ("after" if after else "before")
            / "interpretability"
            / "attn_scores"
        )
        Path.mkdir(attn_scores_subdir, exist_ok=True, parents=True)

        for part in task_data.parts:
            if after:
                part_result = part.result_after.interpretability.result
            else:
                part_result = part.result_before.interpretability.result

            assert type(part_result) is dict

            try:
                file_name = (
                    f"attn_scores-{part.task_id}-{part.sample_id}-{part.part_id}.txt"
                )
                if part_result["attn_scores"]:
                    attn_scores = [
                        "\t".join(map(str, row))
                        for row in part_result["attn_scores"].tolist()
                    ]
                    self.save_with_separator(
                        file_path=attn_scores_subdir / file_name, data=attn_scores
                    )
                for tokens in ("x_tokens", "y_tokens"):
                    if not part_result[tokens]:
                        continue
                    file_name = (
                        f"{tokens}-{part.task_id}-{part.sample_id}-{part.part_id}.txt"
                    )
                    self.save_with_separator(
                        file_path=attn_scores_subdir / file_name,
                        data=part_result[tokens],
                    )
            except AttributeError as e:
                print(f"AttributeError: {e} in {part_result}")

        print(
            f"Interpretability results for task {task_data.task_id} saved to {attn_scores_subdir}"
        )

    def save_task_result(
        self,
        task_id: int,
        task_data: Task,
        headers: list[str],
        results_file_name: str,
        metrics_file_name: str,
        multi_system: bool = False,
    ) -> None:
        """
        Save the results for the task and the accuracy for the task to the separate files.

        :param task_id: the task id
        :param task_data: the result of the task
        :param headers: the headers for the results
        :param results_file_name: the name of the file to save the results specific to the split
        :param metrics_file_name: the name of the file to save the accuracy specific to the split
        :param multi_system: if the setting uses two models

        :return: None
        """
        self.save_output(
            data=task_data.results,
            headers=headers,
            file_name=results_file_name,
        )
        # get accuracy for the last task
        task_accuracy = {"task_id": task_id, **task_data.evaluator_after.get_metrics()}

        if multi_system:
            task_accuracy.update(**task_data.evaluator_before.get_metrics())
            self.save_interpretability(task_data, after=False)

        self.save_output(
            data=[task_accuracy],
            headers=list(task_accuracy.keys()),
            file_name=metrics_file_name,
            path_add="",
        )
        self.save_interpretability(task_data, after=True)

    def save_run_accuracy(
        self,
        task_ids: list[int],
        splits: dict[Prompt, Split],
        features: Features,
        split_name: str,
        after: bool = True,
    ) -> None:
        """
        Save the accuracies for the split run, including the mean accuracy for all tasks.

        :param task_ids: the task ids
        :param splits: the split objects of the data
        :param features: the features to save
        :param split_name: the name of the split
        :param after: if to save the accuracy for after the setting was applied
        :return: None
        """
        run_metrics = {}
        run_headers = ["task_id"]

        for prompt, split in splits.items():
            prompt_headers = prepare_accuracy_headers(prompt.name, after=after)

            if after:
                evaluator = split.evaluator_after
            else:
                evaluator = split.evaluator_before

            run_metrics = format_task_accuracies(
                accuracies_to_save=run_metrics,
                task_ids=task_ids,
                exact_match_accuracies=evaluator.exact_match_accuracy,
                soft_match_accuracies=evaluator.soft_match_accuracy,
                exact_match_std=evaluator.exact_match_std,
                soft_match_std=evaluator.soft_match_std,
                headers=prompt_headers,
            )

            run_metrics = format_split_metrics(
                features, prompt_headers, run_metrics, after=True if after else False
            )
            run_headers.extend(prompt_headers.values())

        mean_headers = prepare_accuracy_headers("mean")
        run_headers.extend(mean_headers.values())
        run_metrics = calculate_mean_accuracies(run_metrics, mean_headers, after=after)

        self.save_output(
            data=list(run_metrics.values()),
            headers=run_headers,
            file_name=self.run_path / f"{split_name}_accuracies.csv",
            path_add="after" if after else "before",
        )

    def redirect_printing_to_log_file(self, file_name: str) -> TextIO:
        """
        Allows to redirect printing during the script run from console into a log file.
        Old 'sys.stdout' that must be returned in place after the run by calling
        DataHandler.return_console_printing!

        :param file_name: the path and name of the file to redirect the printing
        :return: log file to write into
        """
        log_file = open(self.results_path / file_name, "w", encoding="UTF-8")
        sys.stdout = log_file
        return log_file

    def return_console_printing(self, file_to_close: TextIO) -> None:
        """
        This function allows to return the console printing to the original state.

        :param file_to_close: the file that should be closed
        :return:
        """
        file_to_close.close()
        sys.stdout = self.old_stdout
