from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import TextIO
from datetime import datetime

from data.data_utils import is_empty_file


class DataSaver:
    """
    This class handles everything related to saving data.
    """

    def __init__(self, results_path: Path = Path("./")) -> None:
        """
        Initialize the DataSaver.
        The datasaver handles everything related to saving data.
        """
        self.old_stdout: TextIO = sys.stdout
        self.results_path = results_path
        self.run = self.compose_run_name()

    @staticmethod
    def compose_run_name() -> str:
        """
        Compose the name for the run based on the current date and time.
        Example: '2025-01-14_21-48-48'

        :return: the name of the run
        """
        date = str(datetime.now()).split(".")[0]
        date = date.replace(" ", "_")
        date = date.replace(":", "-")
        return date

    def create_path_files(self, results_path: Path, prompt_name: str) \
            -> tuple[Path, Path, dict[str, Path]]:
        """
        Create the unique results path for the run and the files to save the data:
        log file, results file, and accuracy files.

        :param results_path: the path to the results directory
        :param prompt_name: the name of the prompt
        :return: the paths to the log file, results file, and accuracy files in a tuple
        """
        self.results_path = results_path / prompt_name / self.run

        try:
            os.makedirs(self.results_path)
        except FileExistsError:
            print(f"Directory {self.results_path} already exists and is not empty. "
                  "Please choose another results_path or empty the directory.")
            self.results_path = Path("./results") / prompt_name / self.run
            os.makedirs(self.results_path)

        print(f"Saving the results to {self.results_path}")
        
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
