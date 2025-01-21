from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO, Union

from baseline.config.baseline_config import DataSplits
from data.data_utils import is_empty_file


class DataSaver:
    """
    This class handles everything related to saving data.
    """

    def __init__(
        self,
        project_dir: str,
        subproject_dir: str = "",
        save_to_repo: bool = False,
    ) -> None:
        """
        Initialize the DataSaver.
        The datasaver handles everything related to saving data.
        """
        self.old_stdout: TextIO = sys.stdout
        self.run_results_path = Path(project_dir)

        if save_to_repo:
            self.run = self.compose_run_name()
            self.run_results_path = self.compose_run_path(
                project_dir=project_dir, subproject_dir=subproject_dir
            )

    @staticmethod
    def compose_run_name() -> Path:
        """
        Compose the name for the run based on the current date and time.
        Example: ['2025-01-14', '21-48-48']

        :return: date and time of the run in a list
        """
        date_time = str(datetime.now()).split(".")[0]
        date_time = date_time.split()
        date, time = date_time[0], date_time[1].replace(":", "-")
        return Path(date) / time

    def compose_run_path(self, project_dir: str, subproject_dir: str) -> Path:
        """
        Compose the unique results path for the run.

        :param project_dir: the name of the project
        :param subproject_dir: the name of the subproject
        :return: the path to the run
        """
        self.run_results_path = Path(project_dir) / subproject_dir / self.run
        return self.run_results_path

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
        prompt_results_path = self.run_results_path / prompt_name

        try:
            os.makedirs(prompt_results_path)
        except FileExistsError:
            print(
                f"Directory {prompt_results_path} already exists and is not empty. "
                "Please choose another results_path or empty the directory."
            )
            prompt_results_path = Path("./results") / self.run / prompt_name
            os.makedirs(prompt_results_path)
        except OSError:
            print(
                f"Creation of the directory {prompt_results_path} failed due "
                f"to lack of writing rights. Please check the path."
            )
            prompt_results_path = Path("./results") / self.run / prompt_name
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
