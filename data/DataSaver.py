from __future__ import annotations

import csv
import statistics
import sys
from typing import TextIO, Union

from data.utils import *
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
        self.results_path = Path(save_to)
        self.run = self.results_path.parent

    def create_result_paths(
        self,
        prompt_name: str,
        splits: list[Union[DataSplits.train, DataSplits.valid, DataSplits.test]],
    ) -> tuple[Path, dict[str, Path], dict[str, Path]]:
        """
        Create the unique results path for the run and the files to save the data:
        log file, results file, and accuracy files.

        :param prompt_name: the name of the prompt
        :param splits: splits of the data
        :return: the paths to the log file, results file, and accuracy files in a tuple
        """
        prompt_results_path = self.results_path / prompt_name

        try:
            os.makedirs(prompt_results_path)
        except FileExistsError:
            print(
                f"Directory {prompt_results_path} already exists and is not empty. "
                "Please choose another results_path or empty the directory."
            )
            prompt_results_path = Path("./outputs") / prompt_results_path
            os.makedirs(prompt_results_path)
        except OSError:
            print(
                f"Creation of the directory {prompt_results_path} failed due "
                f"to lack of writing rights. Please check the path."
            )
            prompt_results_path = Path("./outputs") / prompt_results_path
            os.makedirs(prompt_results_path)

        print(f"\nThe results will be saved to {prompt_results_path}\n")

        results_file_paths = {}
        accuracy_file_paths = {}

        for split in splits:
            results_file_paths[split] = (
                prompt_results_path / f"{split}_{prompt_name}_results.csv"
            )
            accuracy_file_paths[split] = (
                prompt_results_path / f"{split}_{prompt_name}_accuracies.csv"
            )

        log_file_path = prompt_results_path / f"{prompt_name}.log"

        return log_file_path, results_file_paths, accuracy_file_paths

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
        with open(file_path, "a+", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t")
            if is_empty_file(file_path):
                writer.writeheader()
            [writer.writerow(row) for row in data]

    def save_task_accuracies(
        self,
        task_ids: list[int],
        strict_accuracies: list[float],
        soft_match_accuracies: list[float],
        file_path: Path,
    ) -> [float, float]:
        """
        Save the accuracies for the split,
        including the mean accuracy for all tasks if get_mean_accuracies() was called.

        :param task_ids: the task ids (including 0 if mean accuracies are calculated)
        :param strict_accuracies: the accuracies per task
        :param soft_match_accuracies: the soft match accuracies per task
        :param file_path: the path to the file to save the accuracies
        :return: the mean accuracies of all tasks
        """
        accuracies_to_save = []
        zipped_data = zip(task_ids, strict_accuracies, soft_match_accuracies)
        for task_id, accuracy, soft_match_accuracy in zipped_data:
            accuracies_to_save.append(
                {
                    "task_id": "mean" if task_id == 0 else task_id,
                    "accuracy": accuracy,
                    "soft_match_accuracy": soft_match_accuracy,
                }
            )
        # Save the prompt accuracies for the split
        self.save_output(
            data=accuracies_to_save,
            headers=["task_id", "accuracy", "soft_match_accuracy"],
            file_path=file_path,
        )

    def save_run_accuracies(
        self,
        task_ids: list[int],
        strict_accuracies: dict[str, dict],
        soft_match_accuracies: dict[str, dict],
        split: Union[DataSplits.train, DataSplits.valid, DataSplits.test],
    ) -> None:
        """
        Save the accuracies for the run, including the mean accuracy for all tasks.

        :param task_ids: the task ids (including 0 if mean accuracies are calculated)
        :param strict_accuracies: the strict accuracies per split and prompt
        :param soft_match_accuracies: the soft match accuracies per split and prompt
        :param split: the split of the data
        :return: None
        """
        run_accuracies = {}
        run_headers = ["task_id"]

        split_data = zip(
            strict_accuracies[split].keys(),
            strict_accuracies[split].values(),
            soft_match_accuracies[split].values(),
        )
        # calculate accuracies of each prompt (for all tasks)
        for prompt_name, strict_accuracies, soft_match_accuracies in split_data:
            prompt_headers = prepare_accuracy_headers(prompt_name)
            # by index 0 are the mean accuracies of all tasks
            prompt_data = zip(task_ids, strict_accuracies, soft_match_accuracies)

            for task_id, (strict_acc, soft_match_acc) in prompt_data:
                run_accuracies = add_accuracies(
                    accuracies=run_accuracies,
                    strict_acc_to_add=strict_acc,
                    soft_match_acc_to_add=soft_match_acc,
                    task_id="mean" if task_id == 0 else task_id,
                    headers=prompt_headers,
                )
            run_headers.extend(prompt_headers.values())

        mean_headers = prepare_accuracy_headers("mean")
        run_headers.extend(mean_headers.values())

        # calculate mean accuracies of all tasks (for all prompts)
        for task_id, accuracies in run_accuracies.items():
            strict_accuracies_ = [
                value for name, value in accuracies.items() if "strict" in name
            ]
            soft_match_accuracies_ = [
                value for name, value in accuracies.items() if "soft_match" in name
            ]
            mean_strict_acc_ = round(statistics.mean(strict_accuracies_), 2)
            mean_soft_match_acc_ = round(statistics.mean(soft_match_accuracies_), 2)
            run_accuracies = add_accuracies(
                accuracies=run_accuracies,
                strict_acc_to_add=mean_strict_acc_,
                soft_match_acc_to_add=mean_soft_match_acc_,
                task_id=task_id,
                headers=mean_headers,
            )

        self.save_output(
            data=list(run_accuracies.values()),
            headers=run_headers,
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
        log_file = open(file_name, "w")
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
