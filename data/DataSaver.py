from __future__ import annotations

import csv
import sys
from typing import TextIO, Union

from data.utils import *
from evaluation.Evaluator import MetricEvaluator
from inference.DataLevels import Split, SamplePart
from inference.Prompt import Prompt
from settings.config import DataSplits


class DataSaver:
    """
    This class handles everything related to saving data.
    """

    def __init__(
        self,
        save_to: str,
    ) -> None:
        """
        Initialize the DataSaver.
        The datasaver handles everything related to saving data.

        :param save_to: the path to save the data
        """
        self.old_stdout: TextIO = sys.stdout
        # self.results_path is updated in create_result_paths
        self.results_path = Path(save_to)
        self.run_path = Path(save_to)

    def create_result_paths(
        self,
        prompt_name: str,
        splits: list[Union[DataSplits.train, DataSplits.valid, DataSplits.test]],
    ) -> tuple[Path, dict[str, Path], dict[str, Path]]:
        """
        Create the unique results path for the run and the files to save the data:
        log file, results file, and metrics files. The results path is also updated to match the current prompt.

        :param prompt_name: the name of the prompt
        :param splits: splits of the data
        :return: the paths to the log file, results file, and metrics files in a tuple
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

        results_file_paths = {}
        metrics_file_paths = {}

        for split in splits:
            results_file_paths[split] = (
                self.results_path / f"{split}_{prompt_name}_results.csv"
            )
            metrics_file_paths[split] = (
                self.results_path / f"{split}_{prompt_name}_metrics.csv"
            )

        log_file_path = self.results_path / f"{prompt_name}.log"

        return log_file_path, results_file_paths, metrics_file_paths

    @staticmethod
    def save_output(
        data: list[dict[str, str | int | float]], headers: list | tuple, file_path: Path
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
        :param file_path: the name of the file to save the data
        :return: None
        """
        if file_path.suffix != ".csv":
            raise ValueError("The file should be saved in a .csv format.")

        with open(file_path, "a+", encoding="UTF-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t")
            if is_empty_file(file_path):
                writer.writeheader()
            [writer.writerow(row) for row in data]

    def save_split_accuracy(
        self,
        evaluator: MetricEvaluator,
        accuracy_path: Path,
    ) -> None:
        """
        Save the accuracies for the split,
        including the mean accuracy for all tasks.

        :param evaluator: the evaluator
        :param accuracy_path: the path to the file to save the accuracies
        :return: None
        """
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
            file_path=accuracy_path,
        )

    def save_split_metrics(self, data: Split, results_paths: list[Path]) -> None:
        """
        Save the metrics for all the tasks in a split.

        :param data: the prompt data level
        :param results_paths: the path to save the results
        :return: None
        """
        headers = ["id", "task_id"]
        data = [
            {h: m for h, m in zip(headers, metric)}
            for metric in data.features.get().items()
        ]
        for results_path in results_paths:
            self.save_output(
                data=data,
                headers=headers,
                file_path=results_path,
            )

    def save_interpretability(self, sample_part: SamplePart):
        """
        Save the interpretability result.

        :param sample_part: the sample part
        """
        interpretability_result = sample_part.interpretability.get()
        pass

    def save_task_result(
        self,
        task_id: int,
        task_result: list[dict],
        task_evaluator: MetricEvaluator,
        headers: list[str],
        results_path: Path,
        metrics_path: Path,
    ) -> None:
        """
        Save the results for the task and the accuracy for the task to the separate files.

        :param task_id: the task id
        :param task_result: the result of the task
        :param task_evaluator: the evaluator for the task
        :param headers: the headers for the results
        :param results_path: the path to save the results
        :param metrics_path: the path to save the accuracy
        :return: None
        """
        self.save_output(
            data=task_result,
            headers=headers,
            file_path=results_path,
        )
        # get accuracy for the last task
        task_accuracy = {
            "task_id": task_id,
            "exact_match_accuracy": task_evaluator.exact_match_accuracy.get_mean(),
            "soft_match_accuracy": task_evaluator.soft_match_accuracy.get_mean(),
            "exact_match_std": task_evaluator.exact_match_accuracy.get_std(),
            "soft_match_std": task_evaluator.soft_match_accuracy.get_std(),
        }
        self.save_output(
            data=[task_accuracy],
            headers=list(task_accuracy.keys()),
            file_path=metrics_path,
        )

    def save_run_accuracy(
        self,
        task_ids: list[int],
        prompt_evaluators: dict[Prompt, MetricEvaluator],
        split: Split,
    ) -> None:
        """
        Save the accuracies for the split run, including the mean accuracy for all tasks.

        :param task_ids: the task ids
        :param prompt_evaluators: the evaluators per prompt
        :param split: the split data
        :return: None
        """
        run_metrics = {}
        run_headers = ["task_id"]

        for prompt, evaluator in prompt_evaluators.items():
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
            run_metrics = format_split_metrics(split, prompt_headers, run_metrics)
            run_headers.extend(prompt_headers.values())

        mean_headers = prepare_accuracy_headers("mean")
        run_headers.extend(mean_headers.values())
        run_metrics = calculate_mean_accuracies(run_metrics, mean_headers)

        self.save_output(
            data=list(run_metrics.values()),
            headers=run_headers,
            file_path=self.results_path / f"{split.name}_accuracies.csv",
        )

    @staticmethod
    def redirect_printing_to_log_file(file_name: Path) -> TextIO:
        """
        Allows to redirect printing during the script run from console into a log file.
        Old 'sys.stdout' that must be returned in place after the run by calling
        DataHandler.return_console_printing!

        :param file_name: the path and name of the file to redirect the printing
        :return: log file to write into
        """
        log_file = open(file_name, "w", encoding="UTF-8")
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
