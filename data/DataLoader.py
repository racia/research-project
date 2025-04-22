from __future__ import annotations

import csv
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np

from data.DataProcessor import DataProcessor
from data.utils import get_real_value, structure_parts, get_samples_per_task
from inference.DataLevels import SamplePart
from interpretability.utils import InterpretabilityResult
from settings.config import DataSplits, Enumerate, Wrapper


class SilverReasoning:
    def __init__(self, loader: DataLoader = None):
        """
        Initialize the SilverReasoning class.
        This class handles the loading and retrieval of silver reasoning data.
        """
        self.silver_reasoning_data: dict[tuple[int, int, int], dict[str, str]] = {}
        self.split = ""
        self.task_id = 0
        self.loader = loader

    def get(
        self,
        task_id: int | list[int],
        sample_id: int = None,
        part_id: int = None,
        split: str = "valid",
        from_zero: bool = False,
        get_all: bool = False,
    ) -> str | dict[tuple[int, int, int], dict[str, str]]:
        """
        Get the silver reasoning for the part.
        This method load the data for the whole task and only reloads it if the
        task_id or split changes. If the task_id is a list, it loads the data for
        all tasks and returns a dictionary with the reasoning for each task.

        :param task_id: The task id.
        :param sample_id: The sample id (not used if task_id is a list).
        :param part_id: The part id (not used if task_id is a list).
        :param split: The split of the data.
        :param from_zero: Whether the part ids start from zero.
        :param get_all: Whether to return all reasoning data.
        :return: The silver reasoning for the part.
        """
        if isinstance(task_id, int):
            # reload the data if task_id or split changes
            if task_id != self.task_id or split != self.split:
                self.task_id = task_id
                self.split = split
                self.silver_reasoning_data = self.loader.load_reasoning_data(
                    task_id=task_id, split=split
                )
            # enable getting all reasoning data for a specific task (not only one part)
            if get_all:
                assert type(self.silver_reasoning_data) == dict
                return self.silver_reasoning_data

        # enable getting all reasoning data for a list of tasks
        elif isinstance(task_id, list):
            all_reasoning = {}
            for t in task_id:
                task_reasoning = self.get(
                    t, split=split, from_zero=from_zero, get_all=True
                )
                all_reasoning.update(task_reasoning)
            return all_reasoning

        if not (sample_id or part_id):
            raise ValueError(
                "Either sample_id or part_id must be provided when loading a specific reasoning case."
            )

        if from_zero:
            sample_id, part_id = sample_id - 1, part_id - 1

        assert type(task_id) is int

        # get the reasoning for the specific part
        row = self.silver_reasoning_data.get((task_id, sample_id, part_id), None)
        if row:
            return row["silver_reasoning"]

        available_keys = list(self.silver_reasoning_data.keys())
        raise ValueError(
            f"Silver reasoning for <task_id={task_id}, sample_id={sample_id}, part_id={part_id}> not found."
            f"Available keys: {available_keys}"
        )


class DataLoader:
    """
    This class handles loading the data.
    """

    def __init__(
        self,
        samples_per_task: int = None,
        prefix: str | Path = "",
        wrapper: Wrapper = None,
        to_enumerate: Enumerate = None,
    ):
        """
        Initialize the DataLoader.
        The dataloader handles the reading and loading of data, as well as the mapping of tasks to
        their respective data.

        :param samples_per_task: number of samples to load per task
        :param prefix: path to the data
        :param wrapper: wrapper for the data
        :param to_enumerate: enumeration for the data
        """
        self.number_of_parts: int = 0
        self.samples_per_task: int = samples_per_task
        self.number_of_tasks: int = 0
        self.tasks: list[int] = []

        self.prefix: Path = Path(prefix)
        self.silver_reasoning_path: Path = self.prefix / "data/silver_reasoning/"

        self.wrapper: Wrapper = wrapper
        self.to_enumerate: Enumerate = to_enumerate

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
        multi_system: bool,
        tasks: list[int] = None,
        flat: bool = False,
        lookup: bool = False,
    ) -> (
        list[SamplePart]
        | dict[tuple, SamplePart]
        | dict[int, dict[int, list[SamplePart]]]
    ):
        """
        Prepare the data: load raw data and process it.

        :param path: path to the data
        :param split: should be of type DataSplits ("train", "valid", or "test")
        :param multi_system: if as_parts is True, the multi_system parameter should be specified
        :param tasks: list of task numbers to read, use None to read all tasks
        :param flat: if the data should be returned flat or in a nested structure of tasks and samples
        :param lookup: if the data should be returned as a lookup dictionary
        :return: processed data
        """
        processor = DataProcessor(wrapper=self.wrapper, to_enumerate=self.to_enumerate)
        self.tasks = list(tasks) if bool(tasks) else list(range(1, 21))
        raw_data = self.load_raw_task_data(
            path=Path(path), split=split, tasks=self.tasks
        )
        silver_reasoning = SilverReasoning(self)
        reasoning = silver_reasoning.get(task_id=self.tasks, split=split)
        processed_data: list[SamplePart] = processor.process_data(
            raw_data,
            self.samples_per_task,
            multi_system=multi_system,
            silver_reasoning=reasoning,
        )
        if self.number_of_tasks == 0:
            self.number_of_tasks = len(self.tasks)
        if self.number_of_tasks == 0:
            self.number_of_parts = len(processed_data)

        if flat and lookup:
            raise ValueError(
                "The 'flat' and 'lookup' parameters cannot be used together."
            )
        if flat:
            return processed_data

        if not lookup:
            return structure_parts(processed_data)

        lookup_data = {}
        for part in processed_data:
            lookup_data[(part.task_id, part.sample_id, part.part_id)] = part

        return lookup_data

    def load_results(
        self,
        results_path: str | Path,
        data_path: str | Path = None,
        headers: list[str] = None,
        list_output: bool = False,
        as_parts: bool = False,
        split: str = None,
        tasks: list[int] | None = None,
        sep: str = "\t",
    ) -> (
        dict[str, list[Union[int, float]]]
        | dict[int, list[dict[str, Union[int, float, str]]]]
        | dict[int, dict[int, list[SamplePart]]]
    ):
        """
        Load the results or any csv file from the path.
        Please specify the headers to ensure the desired order of the data.

        :param results_path: path to the results, if None, the data_path is used
        :param data_path: path to the source data
        :param headers: list of headers in the file, if None, the headers are read from the file
        :param list_output: if the output should be a list of dictionaries instead of a dictionary of lists
        :param as_parts: if the output should be a list of SamplePart objects
        :param split: split of the data (to find the file)
        :param tasks: list of task ids to load
        :param sep: separator for the csv file
        :return: result data
        """
        if list_output and as_parts:
            raise ValueError(
                "The 'list_output' and 'as_parts' parameters cannot be used together."
            )
        path = Path(results_path)
        if not path.is_file():
            for p in path.iterdir():
                if split in p.name and p.name.endswith(".csv"):
                    path = p
                    break

        data = [] if list_output or as_parts else defaultdict(list)
        with open(path, "rt", encoding="UTF-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=sep)
            printed = []
            if not headers:
                headers = reader.fieldnames

            for row in reader:
                # to make sure we are reading a row containing task data, not metrics
                if not row[list(row.keys())[0]].isdigit():
                    continue
                # to make sure we are reading a row containing data
                if list_output or as_parts:
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

        if tasks is None:
            tasks = list(range(1, 21))

        for i, row in enumerate(data):
            if type(row["task_id"]) is not int:
                raise ValueError(
                    f"Task id in row {i} is not an integer: {row['task_id']}"
                )
            if row["task_id"] not in tasks:
                data.pop(i)

        self.number_of_parts = len(data)
        self.number_of_tasks = len(tasks)

        if as_parts:
            parts = []
            self.samples_per_task = get_samples_per_task(data)
            raw_parts = self.load_task_data(
                path=data_path,
                split=split,
                tasks=tasks,
                multi_system=True,
                lookup=True,
            )
            for row in data:
                raw_part = raw_parts[(row["task_id"], row["sample_id"], row["part_id"])]
                if not row["model_output_before"]:
                    raise ValueError(
                        f"Model output before is not found in row {row['id_']}: {row['model_output_before']}"
                    )
                raw_part.set_output(
                    model_output=str(row["model_output_before"]),
                    model_answer=str(row["model_answer_before"]),
                    model_reasoning=str(row["model_reasoning_before"]),
                    interpretability=None,
                    full_task=row["task"],
                    version="before",
                )
                parts.append(raw_part)

            if len(parts) != self.number_of_parts:
                warnings.warn(
                    "The number of parts does not match the number of loaded data: %d != %d"
                    % (len(parts), self.number_of_parts)
                )
            return structure_parts(parts)

        return data

    @staticmethod
    def load_scenery(
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
        Additionally, adds Scenery base phrasal words.

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
                    Path.cwd() / path, list_output=True, sep=",", as_parts=False
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
        if not path.exists():
            warnings.warn(
                f"The interpretability data is not found at path: {attn_scores_path}"
            )
            return InterpretabilityResult(np.array(0), [], [], 0)

        if path.name != "attn_scores":
            raise ValueError("The attention subdirectory is not located.")

        try:
            attn_scores_file = f"attn_scores-{task_id}-{sample_id}-{part_id}.txt"
            with open(path / attn_scores_file, "r", encoding="UTF-8") as f:
                attn_scores_rows = f.read().splitlines()
                attn_scores = [
                    list(map(float, row.split("\t"))) for row in attn_scores_rows
                ]
        except FileNotFoundError:
            attn_scores = []

        try:
            x_tokens_file = f"x_tokens-{task_id}-{sample_id}-{part_id}.txt"
            with open(path / x_tokens_file, "r", encoding="UTF-8") as f:
                x_tokens = [token.strip() for token in f.read().splitlines()]
        except FileNotFoundError:
            x_tokens = []

        try:
            y_tokens_file = f"y_tokens-{task_id}-{sample_id}-{part_id}.txt"
            with open(path / y_tokens_file, "r", encoding="UTF-8") as f:
                y_tokens = [token.strip() for token in f.read().splitlines()]
        except FileNotFoundError:
            y_tokens = []

        interpretability_result = InterpretabilityResult(
            attn_scores=np.array(attn_scores),
            x_tokens=x_tokens,
            y_tokens=y_tokens,
        )
        if interpretability_result.empty():
            warnings.warn(
                f"The interpretability data is not found for task {task_id}, "
                f"sample {sample_id}, part {part_id}."
            )
        return interpretability_result


if __name__ == "__main__":
    data_loader = DataLoader(samples_per_task=5)
    data = data_loader.load_task_data(
        path="../../tasks_1-20_v1-2/en-valid/",
        split=DataSplits.valid,
        multi_system=False,
    )
    print(data_loader.number_of_parts)
