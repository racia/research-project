from __future__ import annotations

import csv
import sys
from typing import TextIO, Union

from data.utils import *
from evaluation.Evaluator import MetricEvaluator
from prompts.Prompt import Prompt
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
        with open(file_path, "a+", encoding="UTF-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t")
            if is_empty_file(file_path):
                writer.writeheader()
            [writer.writerow(row) for row in data]


    def save_fine_tune_data(self, task_id: int, task_data: list):
    # Save decoded output to task_id file for fine-tuning
        with open(f"{self.results_path/str(task_id)}.txt", "a") as f:
            for part in task_data:
                f.write("\n".join(part["part"].split("\n\n"))) # Part
                f.write(part["model_reasoning"]) # Only Model Reason - .split("\n\n")[0].split("Reason: ")[-1]
                f.write("\n\n") # Add new line at the end


    def save_task_accuracy(
        self,
        evaluator: MetricEvaluator,
        accuracy_path: Path,
    ) -> [float, float]:
        """
        Save the accuracies for the split,
        including the mean accuracy for all tasks

        :param evaluator: the evaluator
        :param accuracy_path: the path to the file to save the accuracies
        :return: the mean accuracies of all tasks
        """
        accuracies_to_save = list(
            format_accuracy_metrics(
                evaluator.exact_match_accuracy, evaluator.soft_match_accuracy
            ).values()
        )
        headers = list(accuracies_to_save[0].keys())
        self.save_output(
            data=accuracies_to_save,
            headers=headers,
            file_path=accuracy_path,
        )

    def save_task_metrics(
        self, evaluator: MetricEvaluator, results_paths: list[Path]
    ) -> None:
        """
        Save the metrics for the task.

        :param evaluator: the evaluator
        :param results_paths: the path to save the results
        :return: None
        """
        headers = ["id", "task_id"]
        metrics = {
            "there": evaluator.there,
            "verbs": evaluator.verbs,
            "pronouns": evaluator.pronouns,
            "not_mentioned": evaluator.not_mentioned,
        }
        data = [{h: m for h, m in zip(headers, metric)} for metric in metrics.items()]
        for results_path in results_paths:
            self.save_output(
                data=data,
                headers=headers,
                file_path=results_path,
            )

    def save_task_result(
        self, task_id, task_result, task_evaluator, headers, results_path, metrics_path
    ):
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
            "exact_match_accuracy": task_evaluator.exact_match_accuracy[-1],
            "soft_match_accuracy": task_evaluator.soft_match_accuracy[-1],
        }
        self.save_output(
            data=[task_accuracy],
            headers=list(task_accuracy.keys()),
            file_path=metrics_path,
        )

    def save_split_accuracy(
        self,
        task_ids: list[int],
        prompt_evaluators: dict[Prompt, MetricEvaluator],
        split: DataSplits,
    ) -> None:
        """
        Save the accuracies for the run, including the mean accuracy for all tasks.

        :param task_ids: the task ids
        :param prompt_evaluators: the evaluators per prompt
        :param split: the split of the data
        :return: None
        """
        split_metrics = {}
        split_headers = ["task_id"]

        for prompt, evaluator in prompt_evaluators.items():
            prompt_headers = prepare_accuracy_headers(prompt.name)
            split_metrics = format_task_accuracies(
                accuracies_to_save=split_metrics,
                task_ids=task_ids,
                exact_match_accuracies=evaluator.exact_match_accuracy,
                soft_match_accuracies=evaluator.soft_match_accuracy,
                headers=prompt_headers,
            )
            split_metrics = format_task_metrics(
                evaluator, prompt_headers, split_metrics
            )
            split_headers.extend(prompt_headers.values())

        mean_headers = prepare_accuracy_headers("mean")
        split_headers.extend(mean_headers.values())
        split_metrics = calculate_mean_accuracies(split_metrics, mean_headers)

        self.save_output(
            data=list(split_metrics.values()),
            headers=split_headers,
            file_path=self.results_path / f"{split}_accuracies.csv",
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
