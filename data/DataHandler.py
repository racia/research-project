from __future__ import annotations

import csv
import os
import sys
import re
from pathlib import Path
from typing import Dict, TextIO, List, Tuple, Union

from baseline.config.baseline_config import CSVHeaders, DataSplits


class DataHandler:
    def __init__(self):
        """Class to handle data preprocessing."""
        self.task_map = {}
        # very hard to calculate when in another place
        self.question_counter = 0

        self.old_stdout: TextIO = sys.stdout

        self.results_path: str = ""
        self.results_headers: List[str] = []

    def get_task_mapping(self, path: str) -> None:
        """
        Get the paths for each task.

        :param path: path to the data
        :return: None
        """
        path = os.path.abspath(path)
        for dir_path, dir_names, files in os.walk(path):
            for file in files:
                if file.startswith("qa"):
                    task_num = int(file.split("_")[0][2:])
                    if task_num not in self.task_map.keys():
                        self.task_map[task_num] = []
                    self.task_map[task_num].append(os.path.join(dir_path, file))

    def read_file(self, file: str) -> dict:
        """
        Read the file.

        :param file: file path
        :return: data = {id: lines}
        """
        data = {}
        line_count = 0
        id_ = 0
        with open(file, "rt", encoding="UTF-8") as f:
            lines = f.readlines()
            for line in lines:
                curr_line_count = int(line.split(" ")[0])
                if curr_line_count == line_count + 1:
                    if id_ not in data.keys():
                        data[id_] = []
                    data[id_].append(line)
                else:
                    id_ += 1
                    if id_ not in data.keys():
                        data[id_] = []
                    data[id_].append(line)
                line_count = curr_line_count
        return data

    def read_data(self, path: str,
                  split: Union[DataSplits.train, DataSplits.valid, DataSplits.test], tasks=None) -> Dict[str, dict]:
        """
        Read data from file.

        :param path: path to the data
        :param split: should be of type DataSplits ("train", "valid", or "test")
        :param tasks: list of task numbers to read
        :return: data = {task_num: {task data}}
        """
        self.get_task_mapping(path)

        all_tasks = {}
        for task, files in self.task_map.items():
            if tasks and task not in tasks:
                continue
            for file in files:
                data_ext = file.split("_")[-1]
                if split in data_ext:
                    all_tasks[task] = self.read_file(file)
                    print(f"File {file} is read.")

        return all_tasks

    def process_data(self, data: Dict[str, dict]) -> dict:
        """
        Process the data.

        :param data: data to process
        :return: processed data of type
                 Dict[str, Dict[str, Dict[str, Dict[str, str] | Dict[str, List[str]] | List[List[int]]]]

                 Example:
                 {
                     task_id: str: {
                         sample_id: str = 0-n: {
                             "context": {
                                 line_num: str
                                 sentence: str
                             }
                             "question": {
                                 line_num: str
                                 question: str
                             }
                             "answer": {
                                 line_num: str
                                 answers: List[str]
                             }
                             "supporting_fact": [
                                 [int], [int, int]
                             ]
                         }
                     }
                 }
        """
        processed_data = {}
        for task in data:
            processed_data[task] = {}
            for id_ in data[task].keys():
                processed_data[task][id_] = {
                    "context": {},
                    "question": {},
                    "answer": {},
                    "supporting_fact": [],
                }
                for line in data[task][id_]:
                    cleaned = line.strip()
                    # regex: group 1: line number: \d+\s+
                    # no group: space: \s+
                    # group 2: question: .+?
                    # no group: space: \s+
                    # group 3: answer: \w+(?:,\w+)?     # there might be two answers (see task 8)
                    # no group: space: \s+
                    # group 4: supporting fact: ((?:\d+\s*)+)
                    question_line_pattern = r"^(\d+)\s+(.+?)\s+(\w+(?:,\w+)?)\s+((?:\d+\s*)+)$"
                    question_match = re.match(question_line_pattern, cleaned)

                    context_line_pattern = r"^(\d+\s+)(.+)$"
                    context_match = re.match(context_line_pattern, cleaned)
                    if question_match:
                        line_num = int(question_match.group(1))
                        processed_data[task][id_]["question"][line_num] = (
                            question_match.group(2)
                        )
                        processed_data[task][id_]["answer"][line_num] = (
                            # there might be two answers
                            question_match.group(3).split(",")
                        )
                        supporting_list = [
                            int(x) for x in question_match.group(4).split(" ")
                        ]
                        processed_data[task][id_]["supporting_fact"].append(
                            supporting_list
                        )
                    elif context_match:
                        line_num = int(context_match.group(1))
                        processed_data[task][id_]["context"][line_num] = (
                            context_match.group(2)
                        )
                    else:
                        print("No match found for line: ", cleaned)
                self.question_counter += len(processed_data[task][id_]["question"])

        return processed_data

    @staticmethod
    def check_or_create_directory(path: str) -> None:
        """
        Check if the directory exists, if not create it.

        :param path: path to the directory
        """
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def is_empty_file(file_path: Path) -> bool:
        """
        Checks if the file exists and is empty.

        :param file_path: the file path to check
        :return: True if file exists and is empty
                 False if file is non-empty
        """
        return os.path.isfile(file_path) and os.path.getsize(file_path) == 0

    def set_results_details(self, results_path: str, headers: List[CSVHeaders]) -> None:
        """Allows to set the path for saving results and headers for the csv file."""
        self.results_path = results_path
        self.results_headers = headers

    def save_output(self, data: List[Dict[str, str | int | float]]) -> None:
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
            writer = csv.DictWriter(file, fieldnames=self.results_headers, delimiter="\t")
            if self.is_empty_file(path):
                writer.writeheader()
            [writer.writerow(row) for row in data]

    def return_console_printing(self, file_to_close):
        file_to_close.close()
        sys.stdout = self.old_stdout

    @staticmethod
    def redirect_printing_to_file(path: str) -> Tuple[TextIO, Path]:
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


if __name__ == "__main__":
    dh = DataHandler()
    data = dh.read_data("../../tasks_1-20_v1-2/en", split="train", tasks=[3])
    processed = dh.process_data(data)
    print(processed)
