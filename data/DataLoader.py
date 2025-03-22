from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np

from data.DataProcessor import DataProcessor
from data.utils import convert_true
from interpretability.utils import InterpretabilityResult
from settings.config import DataSplits


class DataLoader:
    """
    This class handles loading the data.
    """

    def __init__(self, samples_per_task: int = None):
        """
        Initialize the DataLoader.
        The dataloader handles the reading and loading of data, as well as the mapping of tasks to
        their respective data.
        """
        self.number_of_parts: int = 0
        self.samples_per_task: int = samples_per_task
        self.number_of_tasks: int = 0
        self.tasks: list[int] = []

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
        tasks: list[int] = None,
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
        tasks: list[int] = None,
    ) -> dict:
        """
        Prepare the data: load raw data and process it.

        :param path: path to the data
        :param split: should be of type DataSplits ("train", "valid", or "test")
        :param tasks: list of task numbers to read, use None to read all tasks
        :return: processed data
        """
        processor = DataProcessor()
        self.number_of_tasks = len(tasks) if tasks else 20
        self.tasks = tasks if tasks else list(range(1, 21))
        raw_data = self.load_raw_task_data(path=Path(path), split=split, tasks=tasks)
        processed_data = processor.process_data(raw_data, self.samples_per_task)
        self.number_of_parts = processor.part_counter
        if not self.samples_per_task:
            self.samples_per_task = processor.sample_counter
        return processed_data

    @staticmethod
    def load_result_data(
        result_file_path: str,
        headers: list[str],
        list_output: bool = False,
        sep: str = "\t",
    ) -> (
        dict[str, list[Union[int, float]]]
        or dict[int, list[dict[str, Union[int, float, str]]]]
    ):
        """
        Read the accuracy from a file.

        :param result_file_path: path to the file
        :param headers: list of headers in the file
        :param list_output: if the output should be a list of dictionaries or a dictionary of lists
        :param sep: separator for the file
        :return: dictionary with the task, accuracy, and soft match accuracy lists
        """
        with open(Path(result_file_path), "rt", encoding="UTF-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=sep)
            data = [] if list_output else defaultdict(list)
            printed = False
            for row in reader:
                # to make sure we are reading a row containing data
                if row["task_id"].isdigit():
                    if list_output:
                        row_ = {}
                        for header, value in row.items():
                            if header in headers:
                                row_[header] = convert_true(value)
                            else:
                                if not printed:
                                    print(f"Header '{header}' not found in headers.")
                        data.append(row_)
                        printed = True
                    else:
                        for header in headers:
                            if header not in row.keys():
                                print(f"Header '{header}' not found in row.")
                                continue
                            value = row[header]
                            data[header].append(convert_true(value))
        return data

    def load_reasoning_data(
        self, path: str, headers: list[str]
    ) -> dict[tuple[int, int, int], dict]:
        """
        Load the silver reasoning data.

        :param path: path to the silver reasoning data
        :param headers: headers for the silver reasoning data
        """
        silver_reasoning_data = self.load_result_data(
            path,
            headers=headers,
            list_output=True,
            sep=",",
        )
        silver_reasoning_data = {
            (int(row["task_id"]), int(row["sample_id"]), int(row["part_id"])): row
            for row in silver_reasoning_data
        }
        return silver_reasoning_data

    def load_scenery(
        self,
        word_types: tuple[str, ...] = (
            "attr",
            "loc",
            "nh-subj",
            "obj",
            "part",
            "rel",
            "subj-attr",
            "subj",
            "other",
            "base_phrasal_verbs",
        ),
    ) -> set:
        """
        Get scenery words from the scenery_words folder and the Scenery base phrases verbs.
        Additionally adds Scenery base phrasal words.

        :return: set of scenery words for filtering attention scores
        """
        scenery_words = set()
        for entry in os.scandir("data/scenery_words"):
            word_type = entry.name.strip(".txt")
            if word_type in word_types:
                with open(entry.path, "r", encoding="UTF-8") as f:
                    scenery_words.update(f.read().splitlines())
        return scenery_words

    @staticmethod
    def load_interpretability(
        task_id: int, sample_id: int, part_id: int, attn_scores_path: str
    ) -> InterpretabilityResult:
        """
        Load the interpretability results for a specific part.

        :param task_id: task id
        :param sample_id: sample id
        :param part_id: part id
        :param attn_scores_path: path to the attention scores subdirectory
        :return: Interpretability Result object
        """
        path = Path(attn_scores_path)
        if path.name != "attn_scores":
            raise ValueError("The attention subdirectory is not located.")

        attn_scores_file = f"attn_scores-{task_id}-{sample_id}-{part_id}.txt"
        with open(path / attn_scores_file, "r", encoding="UTF-8") as f:
            attn_scores_rows = f.read().splitlines()
            attn_scores = [
                list(map(float, row.split("\t"))) for row in attn_scores_rows
            ]

        x_tokens_file = f"x_tokens-{task_id}-{sample_id}-{part_id}.txt"
        with open(path / x_tokens_file, "r", encoding="UTF-8") as f:
            x_tokens = [token.strip() for token in f.read().splitlines()]

        y_tokens_file = f"y_tokens-{task_id}-{sample_id}-{part_id}.txt"
        with open(path / y_tokens_file, "r", encoding="UTF-8") as f:
            y_tokens = [token.strip() for token in f.read().splitlines()]

        interpretability_result = InterpretabilityResult(
            attn_scores=np.array(attn_scores), x_tokens=x_tokens, y_tokens=y_tokens
        )
        return interpretability_result


if __name__ == "__main__":
    data_loader = DataLoader(samples_per_task=5)
    data = data_loader.load_task_data(
        path="../../tasks_1-20_v1-2/en-valid/",
        split=DataSplits.valid,
    )
    print(data_loader.number_of_parts)
