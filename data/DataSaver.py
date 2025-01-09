from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import TextIO

from data.data_utils import is_empty_file


class DataSaver:
    """
    This class handles everything related to saving data.
    """

    def __init__(self) -> None:
        """
        Initialize the DataSaver.
        The datasaver handles everything related to saving data.
        """
        self.old_stdout: TextIO = sys.stdout
        self.results_path = results_path

    def create_path_files(self, results_path: Path, prompt_name: str, run_number: int) \
            -> tuple[Path, Path, dict[str, Path]]:
        self.results_path = results_path / prompt_name / f"run_{run_number}"

        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        log_file_path = self.results_path / f"{prompt_name}.log"
        results_file_path = self.results_path / f"{prompt_name}_results.csv"

        accuracy_file_paths = {
            "train": self.results_path / f"{prompt_name}_train_accuracies.csv",
            "valid": self.results_path / f"{prompt_name}_valid_accuracies.csv",
            "test": self.results_path / f"{prompt_name}_test_accuracies.csv"
        }

        return log_file_path, results_file_path, accuracy_file_paths

    @staticmethod
    def save_output(data: list[dict[str, str | int | float]],
                    headers: list | tuple, file_path: Path) -> None:
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
            writer = csv.DictWriter(
                file, fieldnames=headers, delimiter="\t"
            )
            if is_empty_file(file_path):
                writer.writeheader()
            [writer.writerow(row) for row in data]

    @staticmethod
    def redirect_printing_to_log_file(file_name) -> TextIO:
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

    def return_console_printing(self, file_to_close):
        """
        This function allows to return the console printing to the original state.

        :param file_to_close: the file that should be closed
        :return:
        """
        file_to_close.close()
        sys.stdout = self.old_stdout
