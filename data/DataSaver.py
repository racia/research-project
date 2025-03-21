from __future__ import annotations

import csv
import sys
from typing import TextIO, Union, Iterable

from data.utils import *
from evaluation.Evaluator import MetricEvaluator
from inference.DataLevels import Task, Features
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
        :return: None
        """
        if isinstance(file_name, str):
            file_name = self.results_path / file_name
            if file_name.suffix != ".csv":
                raise ValueError("The file should be saved in a .csv format.")
        else:
            file_name = Path(file_name)

        with open(file_name, "a+", encoding="UTF-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t")
            if is_empty_file(file_name):
                writer.writeheader()
            [writer.writerow(row) for row in data]

    def save_split_accuracy(
        self,
        evaluator: MetricEvaluator,
        accuracy_file_name: str,
        after: bool = True,
    ) -> None:
        """
        Save the accuracies for the split,
        including the mean accuracy for all tasks.

        :param evaluator: the evaluator
        :param accuracy_file_name: the name of the file to save the accuracies
        :param after: if to save the accuracy for after the setting was applied
        :return: None
        """
        if after:
            accuracy_file_name = self.results_path / "after" / accuracy_file_name
        else:
            accuracy_file_name = self.results_path / "before" / accuracy_file_name

        accuracies_to_save = list(
            format_accuracy_metrics(
                evaluator.exact_match_accuracy,
                evaluator.soft_match_accuracy,
                evaluator.exact_match_std,
                evaluator.soft_match_std,
            ).values()
        )
        headers = list(accuracies_to_save[0].keys())
        self.save_output(
            data=accuracies_to_save,
            headers=headers,
            file_name=accuracy_file_name,
        )

    def save_split_metrics(
        self, features: Features, result_file_names: list[str]
    ) -> None:
        """
        Save the metrics for all the tasks in a split.

        :param features: the features to save
        :param result_file_names: the path to save the results
        :return: None
        """
        headers = ["id", "task_id"]
        features = [
            {h: m for h, m in zip(headers, metric)} for metric in features.get().items()
        ]
        for result_file_name in result_file_names:
            self.save_output(
                data=features,
                headers=headers,
                file_name=result_file_name,
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
            file.write(sep.join(data))

    def save_interpretability(self, task_data: Task, after: bool = True) -> None:
        """
        Save the interpretability result per sample part.

        :param task_data: the task instance with the results
        :param after: if to save the interpretability result for after the setting was applied
        :return: None
        """
        attn_scores_subdir = self.results_path / "interpretability" / "attn_scores"
        Path.mkdir(attn_scores_subdir, exist_ok=True, parents=True)

        for part in task_data.parts:
            if after:
                part_result = part.result_after.interpretability.result
            else:
                part_result = part.result_before.interpretability.result

            try:
                file_name = (
                    f"attn_scores-{part.task_id}-{part.sample_id}-{part.part_id}.txt"
                )
                attn_scores = [
                    "\t".join(map(str, row)) for row in part_result.attn_scores.tolist()
                ]
                self.save_with_separator(
                    file_path=attn_scores_subdir / file_name, data=attn_scores
                )
                for tokens in ("x_tokens", "y_tokens"):
                    file_name = (
                        f"{tokens}-{part.task_id}-{part.sample_id}-{part.part_id}.txt"
                    )
                    self.save_with_separator(
                        file_path=attn_scores_subdir / file_name,
                        data=part_result.tokens,
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
        setting: str,
    ) -> None:
        """
        Save the results for the task and the accuracy for the task to the separate files.

        :param task_id: the task id
        :param task_data: the result of the task
        :param headers: the headers for the results
        :param results_file_name: the name of the file to save the results specific to the split
        :param metrics_file_name: the name of the file to save the accuracy specific to the split
        :param setting: the setting name

        :return: None
        """
        self.save_output(
            data=task_data.results,
            headers=headers,
            file_name=results_file_name,
        )
        # get accuracy for the last task
        task_accuracy = {
            "task_id": task_id,
            "exact_match_accuracy": task_data.evaluator_after.exact_match_accuracy.get_mean(),
            "soft_match_accuracy": task_data.evaluator_after.soft_match_accuracy.get_mean(),
            "exact_match_std": task_data.evaluator_after.exact_match_accuracy.get_std(),
            "soft_match_std": task_data.evaluator_after.soft_match_accuracy.get_std(),
        }

        if setting not in ["SD", "SpeculativeDecoding", "Feedback"]:
            task_accuracy_before = {
                "task_id": task_id,
                "exact_match_accuracy": task_data.evaluator_before.exact_match_accuracy.get_mean(),
                "soft_match_accuracy": task_data.evaluator_before.soft_match_accuracy.get_mean(),
                "exact_match_std": task_data.evaluator_before.exact_match_std.get_mean(),
                "soft_match_std": task_data.evaluator_before.soft_match_std.get_mean(),
            }
            task_accuracy.update(task_accuracy_before)
            self.save_interpretability(task_data, after=False)

        self.save_output(
            data=[task_accuracy],
            headers=list(task_accuracy.keys()),
            file_name=metrics_file_name,
        )
        self.save_interpretability(task_data, after=True)

    def save_run_accuracy(
        self,
        task_ids: list[int],
        split_evaluators: dict[Prompt, MetricEvaluator],
        features: Features,
        split_name: str,
    ) -> None:
        """
        Save the accuracies for the split run, including the mean accuracy for all tasks.

        :param task_ids: the task ids
        :param split_evaluators: the evaluators per prompt
        :param features: the features to save
        :param split_name: the name of the split
        :return: None
        """
        run_metrics = {}
        run_headers = ["task_id"]

        for prompt, evaluator in split_evaluators.items():
            prompt_headers = prepare_accuracy_headers(prompt.name)
            run_metrics = format_task_accuracies(
                accuracies_to_save=run_metrics,
                task_ids=task_ids,
                exact_match_accuracies=evaluator.exact_match_accuracy,
                soft_match_accuracies=evaluator.soft_match_accuracy,
                exact_match_std=evaluator.exact_match_std,
                soft_match_std=evaluator.soft_match_std,
                headers=prompt_headers,
            )
            run_metrics = format_split_metrics(features, prompt_headers, run_metrics)
            run_headers.extend(prompt_headers.values())

        mean_headers = prepare_accuracy_headers("mean")
        run_headers.extend(mean_headers.values())
        run_metrics = calculate_mean_accuracies(run_metrics, mean_headers)

        self.save_output(
            data=list(run_metrics.values()),
            headers=run_headers,
            file_name=self.run_path / f"{split_name}_accuracies.csv",
        )

    @staticmethod
    def redirect_printing_to_log_file(file_name: str) -> TextIO:
        """
        Allows to redirect printing during the script run from console into a log file.
        Old 'sys.stdout' that must be returned in place after the run by calling
        DataHandler.return_console_printing!

        :param file_name: the path and name of the file to redirect the printing
        :return: log file to write into
        """
        log_file = open(Path(file_name), "w", encoding="UTF-8")
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
