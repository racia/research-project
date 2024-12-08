import csv
import os
import re
from typing import List, Dict, Union, Literal


class DataHandler:
    """
    Class to handle data preprocessing.
    """

    DataSplits = Literal["train", "valid", "test"]

    def __init__(self):
        self.task_map = {}
        # would be useful to trace progress on runs when we take all samples per task,
        # otherwise very hard to calculate
        self.question_counter = 0

    def get_task_mapping(self, path) -> None:
        """
        Get the paths for each task.

        Parameters
        ----------
        :param path: path to the data
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

        Parameters
        ----------
        :param file: file path

        Returns
        -------
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

    # into config
    def read_data(self, path: str, split: DataSplits, tasks=None) -> Dict[str, dict]:
        """
        Read data from file.

        Parameters
        ----------
        :param path: path to the data
        :param split: should be of type DataSplits ("train", "valid", or "test")
        :param tasks: list of task numbers to read

        Returns
        -------
        :return: data = {task_num: {task data}}
        """
        self.get_task_mapping(path)

        all_tasks = {}
        for task, files in self.task_map.items():
            if tasks and task not in tasks:
                continue
            for file in files:
                data_ext = file.split("_")[-1]
                # split should be a string of DataSplits["train", "valid", "test"]
                if split in data_ext:
                    all_tasks[task] = self.read_file(file)
                    print(f"File {file} is read.")

        return all_tasks

    def process_data(self, data) -> dict:
        """
        Process the data.

        Parameters
        ----------
        :param data: data to process

        Returns
        -------
        :return: processed data of type
        Dict[str, Dict[str, Dict[str, Union[Dict[str, str], Dict[str, List[str]], List[List[int]]]]]:
        {
            task_id: str
                {
                sample_id: str = 0-n
                    {
                    "context"
                        {
                        line_num: str
                        sentence: str
                        }
                    "question"
                        {
                        line_num: str
                        question: str
                        }
                    "answer"
                        {
                        line_num: str
                        answers: List[str]
                        }
                    "supporting_fact"
                        [
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
    def check_or_create_directory(path: str):
        """
        Check if the directory exists, if not create it.

        :param path: path to the directory
        """
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def is_empty_file(file_path):
        return os.path.isfile(file_path) and os.path.getsize(file_path) == 0

    def save_output(self, path: str, headers: Union[list, tuple], data: list) -> None:
        path = f"{path}.csv"
        # with open(path, "a+", encoding="utf-8") as file:
        #     writer = csv.writer(file, delimiter="\t")
        #     if self.is_empty_file(path):
        #         writer.writerow("\t".join(headers))
        #     if type(data) is List[List]:
        #         writer.writerows(data)
        #     else:
        #         writer.writerow("\t".join(data))

        with open(path, "a+", encoding="utf-8") as file:
            """
            headers = ['first_name', 'last_name']
            row = {'first_name': 'Lovely', 'last_name': 'Spam'}
            writer.writerow(row)
            """
            writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t")
            if self.is_empty_file(path):
                writer.writeheader()
            [writer.writerow(row) for row in data]


if __name__ == "__main__":
    dh = DataHandler()
    data = dh.read_data("data/tasks_1-20_v1-2/en", split="train", tasks=["qa3"])
    processed = dh.process_data(data)
    print(processed)
