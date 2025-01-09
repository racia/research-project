from __future__ import annotations

import os
from typing import Union

from baseline.config.baseline_config import DataSplits
from data.DataProcessor import DataProcessor


class DataLoader:
    """
    This class handles loading the data.
    """

    def __init__(self):
        """
        Initialize the DataLoader.
        The dataloader handles the reading and loading of data, as well as the mapping of tasks to
        their respective data.
        """
        self.task_map = {}

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

    def load_raw_data(
        self,
        path: str,
        split: Union[DataSplits.train, DataSplits.valid, DataSplits.test],
        tasks=None,
    ) -> dict[str, dict]:
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

        sorted_all_tasks = dict(sorted(all_tasks.items(), key=lambda x: int(x[0])))

        return sorted_all_tasks

    def load_data(
            self, path: str,
            split: Union[DataSplits.train, DataSplits.valid, DataSplits.test],
            tasks=None) -> dict:
        """
        Prepare the data: load raw data and process it.

        :param path: path to the data
        :param split: should be of type DataSplits ("train", "valid", or "test")
        :param tasks: list of task numbers to read
        :return: processed data
        """
        processor = DataProcessor()
        raw_data = self.load_raw_data(path=path, split=split, tasks=tasks)
        return processor.process_data(raw_data)
