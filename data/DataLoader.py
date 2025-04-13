from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np

from data.DataProcessor import DataProcessor
from data.utils import get_real_value
from interpretability.utils import InterpretabilityResult
from settings.config import DataSplits


class DataLoader:
    """
    This class handles loading the data.
    """

    def __init__(self, samples_per_task: int = None, prefix: str | Path = ""):
        """
        Initialize the DataLoader.
        The dataloader handles the reading and loading of data, as well as the mapping of tasks to
        their respective data.
        """
        self.number_of_parts: int = 0
        self.samples_per_task: int = samples_per_task
        self.number_of_tasks: int = 0
        self.tasks: list[int] = []

        self.prefix: Path = Path(prefix)
        self.silver_reasoning_path: Path = self.prefix / "data/silver_reasoning/"

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
    def load_results(
        path: str | Path,
        headers: list[str] = None,
        list_output: bool = False,
        sep: str = "\t",
    ) -> (
        dict[str, list[Union[int, float]]]
        or dict[int, list[dict[str, Union[int, float, str]]]]
    ):
        """
        Load the results or any csv file from the path.
        Please specify the headers to ensure the desired order of the data.

        :param path: path to the data
        :param headers: list of headers in the file, if None, the headers are read from the file
        :param list_output: if the output should be a list of dictionaries instead of a dictionary of lists
        :param sep: separator for the csv file
        :return: result data
        """
        data = [] if list_output else defaultdict(list)
        with open(Path(path), "rt", encoding="UTF-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=sep)
            printed = []
            if not headers:
                headers = reader.fieldnames

            for row in reader:
                if not row[list(row.keys())[0]].isdigit():
                    continue
                # to make sure we are reading a row containing data
                if list_output:
                    row_ = {}
                    for header, value in row.items():
                        if header in headers:
                            row_[header] = get_real_value(value)
                        elif header not in printed:
                            print(f"Header '{header}' not found in headers.")
                            printed.append(header)
                    data.append(row_)

                else:
                    for header in headers:
                        if header in row.keys():
                            data[header].append(get_real_value(row[header]))
                        elif header not in printed:
                            print(f"Header '{header}' not found in row.")
                            printed.append(header)

        return data

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

    def load_reasoning_data(
        self, task_id: int, split: str = DataSplits.valid
    ) -> dict[tuple[int, int, int], dict]:
        """
        Load the silver reasoning data for a specific task and split.

        :param task_id: task id
        :param split: split of the data
        :return: silver reasoning data
        """
        if not self.silver_reasoning_path.exists():
            raise FileNotFoundError(
                "The silver reasoning data is not found at the path:",
                self.silver_reasoning_path,
            )
        silver_reasoning_data = []
        for path in self.silver_reasoning_path.iterdir():
            if f"{split}_{task_id}." in path.name:
                silver_reasoning_data = self.load_results(
                    Path.cwd() / path, list_output=True, sep=","
                )
                break

        if not silver_reasoning_data:
            raise FileNotFoundError(
                f"Silver reasoning data for task {task_id} and split {split} is "
                f"not found in the path: {self.silver_reasoning_path}"
            )

        silver_reasoning_data = {
            (int(row["task_id"]), int(row["sample_id"]), int(row["part_id"])): row
            for row in silver_reasoning_data
        }
        return silver_reasoning_data

    def load_interpretability(
        self, task_id: int, sample_id: int, part_id: int, attn_scores_path: str
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

        max_attn_file = f"max_attn-{task_id}-{sample_id}-{part_id}.txt"
        with open(path / max_attn_file, "r", encoding="UTF-8") as f:
            max_attn_dist = json.load(f)

        interpretability_result = InterpretabilityResult(
            attn_scores=np.array(attn_scores),
            x_tokens=x_tokens,
            y_tokens=y_tokens,
            max_attn_dist=max_attn_dist,
        )
        return interpretability_result


if __name__ == "__main__":
    data_loader = DataLoader(samples_per_task=5)
    data = data_loader.load_task_data(
        path="../../tasks_1-20_v1-2/en-valid/",
        split=DataSplits.valid,
    )
    print(data_loader.number_of_parts)
