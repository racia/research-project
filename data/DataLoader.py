from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Union

from baseline.config.baseline_config import DataSplits
from data.DataProcessor import DataProcessor


class DataLoader:
    """
    This class handles loading the data.
    """

    def __init__(self, samples_per_task: int = -1):
        """
        Initialize the DataLoader.
        The dataloader handles the reading and loading of data, as well as the mapping of tasks to
        their respective data.
        """
        self.number_of_parts = 0
        self.samples_per_task = samples_per_task

    @staticmethod
    def get_task_mapping(path: Path) -> dict[int, list[Path]]:
        """
        Get the paths for each task.

        :param path: path to the data
        :return: None
        """
        task_map = {}

        for dir_path, dir_names, files in os.walk(path):
            for file in files:
                if file.startswith("qa"):
                    task_num = int(file.split("_")[0][2:])
                    if task_num not in task_map.keys():
                        task_map[task_num] = []
                    task_map[task_num].append(path / file)

        return task_map

    @staticmethod
    def read_task_file(file: Path) -> dict:
        """
        Read the file for a task.

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

    def load_raw_task_data(
        self,
        path: Path,
        split: Union[DataSplits.train, DataSplits.valid, DataSplits.test],
        tasks=None,
    ) -> dict[int, dict]:
        """
        Read data from file for a split.

        :param path: path to the data
        :param split: should be of type DataSplits ("train", "valid", or "test")
        :param tasks: list of task numbers to read
        :return: data = {task_num: {task data}}
        """
        task_map = self.get_task_mapping(path)

        all_tasks = {}
        for task, files in task_map.items():
            if tasks and task not in tasks:
                continue
            for file in files:
                data_ext = file.stem.split("_")[-1]
                if split in data_ext:
                    all_tasks[task] = self.read_task_file(file)
                    print(f"File {file} is read.")

        sorted_all_tasks = dict(sorted(all_tasks.items(), key=lambda x: int(x[0])))

        return sorted_all_tasks

    def load_task_data(
        self,
        path: str,
        split: Union[DataSplits.train, DataSplits.valid, DataSplits.test],
        tasks: int = None,
    ) -> dict:
        """
        Prepare the data: load raw data and process it.

        :param path: path to the data
        :param split: should be of type DataSplits ("train", "valid", or "test")
        :param tasks: list of task numbers to read, use None to read all tasks
        :return: processed data
        """
        processor = DataProcessor()
        raw_data = self.load_raw_task_data(path=Path(path), split=split, tasks=tasks)
        processed_data = processor.process_data(raw_data, self.samples_per_task)
        self.number_of_parts = processor.part_counter
        return processed_data

    @staticmethod
    def load_result_data(
        result_file_path: str,
        headers: list[str],
        list_output: bool = False,
    ) -> (
        dict[str, list[Union[int, float]]]
        or dict[int, list[dict[str, Union[int, float, str]]]]
    ):
        """
        Read the accuracy from a file.

        :param result_file_path: path to the file
        :param headers: list of headers in the file
        :param list_output: if the output should be a list of dictionaries or a dictionary of lists
        :return: dictionary with the task, accuracy, and soft match accuracy lists
        """
        data = defaultdict(list)
        with open(Path(result_file_path), "rt", encoding="UTF-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if list_output:
                    task_id = int(row["task_id"])
                    row_ = {}
                    for k, v in row.items():
                        if k in headers and v:
                            if v.isdigit():
                                row_[k] = int(v)
                            elif v.replace(".", "", 1).isdigit():
                                row_[k] = float(v)
                            else:
                                row_[k] = v
                    data[task_id].append(row_)
                else:
                    for header in headers:
                        value = row[header]
                        if value.isdigit():
                            data[header].append(int(value))
                        elif value.replace(".", "", 1).isdigit():
                            data[header].append(float(value))
                        else:
                            data[header].append(row[header])
        return data


if __name__ == "__main__":
    data_loader = DataLoader(samples_per_task=5)
    data = data_loader.load_task_data(
        path="../../tasks_1-20_v1-2/en-valid/",
        split=DataSplits.valid,
    )
    print(data_loader.number_of_parts)
