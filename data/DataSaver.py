from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import TextIO

from baseline.config.baseline_config import CSVHeaders
from data_utils import is_empty_file


class DataSaver:
    """
    This class handles everything related to saving data.
    """

    def __init__(self):
        """
        Initialize the DataSaver.
        The datasaver handles everything related to saving data.
        """
        self.old_stdout: TextIO = sys.stdout

        self.results_path: str = ""
        self.results_headers: list[str] = []

    def set_results_details(self, results_path: str, headers: list[CSVHeaders]) -> None:
        """
        Allows to set the path for saving results and headers for the csv file.
        """
        self.results_path = results_path
        self.results_headers = headers

    def save_output(self, data: list[dict[str, str | int | float]]) -> None:
        """
        This function allows to save the data continuously throughout the run.
        The headers are added once at the beginning, and the data is appended
        to the end of the file.

        This is how DictWriter works:
            headers = ['first_name', 'last_name']
            row = {'first_name': 'Lovely', 'last_name': 'Spam'}
            writer.writerow(row)

        :param data: one row as list of strings or multiple such rows
        :return: None
        """
        path = Path(f"{self.results_path}.csv")

        with open(path, "a+", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=self.results_headers, delimiter="\t"
            )
            if is_empty_file(path):
                writer.writeheader()
            [writer.writerow(row) for row in data]

    def return_console_printing(self, file_to_close):
        """
        This function allows to return the console printing to the original state.

        :param file_to_close: the file that should be closed
        :return:
        """
        file_to_close.close()
        sys.stdout = self.old_stdout

    def redirect_printing_to_file(self, path: str) -> tuple[TextIO, Path]:
        """
        Allows to redirect printing during the script run from console into a log file.
        Old 'sys.stdout' that must be returned in place after the run by calling
        DataHandler.return_console_printing!

        :param path: the path to the result directory with a file name (no extension required)
        :return: log fine to write into and Path to the updated result file
        """
        # 'log_file' and 'file_name' will surely be created:
        # if files_1-5 already exist, then a default file_0 would be created/overwritten
        log_file = None
        file_path_name = Path("")

        file_created = False
        for i in range(1, 6):
            file_path_name = Path(f"{path}_{i}")
            if not os.path.isfile(f"{file_path_name}.log"):
                log_file = open(f"{file_path_name}.log", "w")
                file_created = True
                break

        if not file_created:
            file_path_name = Path(f"{path}_0")
            log_file = open(f"{file_path_name}.log", "w")

        sys.stdout = log_file
        return log_file, file_path_name
