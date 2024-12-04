import os
import re


class DataHandler:
    """
    Class to handle data preprocessing.
    """

    def __init__(self):
        self.task_map = {}

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
                    task_num = file.split("_")[0]
                    if not task_num in self.task_map.keys():
                        self.task_map[task_num] = []
                    self.task_map[task_num].append(os.path.join(dir_path, file))

    def read_file(self, file: str) -> dict:
        """
        Read the file.

        Parameters
        ----------
        :param file: file path
        """
        data = {}
        line_count = 0
        id = 0
        with open(file, "rt", encoding="UTF-8") as f:
            lines = f.readlines()
            for line in lines:
                curr_line_count = int(line.split(" ")[0])
                if curr_line_count == line_count + 1:
                    if id not in data.keys():
                        data[id] = []
                    data[id].append(line)
                else:
                    id += 1
                    if id not in data.keys():
                        data[id] = []
                    data[id].append(line)
                line_count = curr_line_count
        return data

    def read_data(self, path: str, task=None, train=False) -> dict:
        """
        Read data from file.

        Parameters
        ----------
        :param path: path to the data
        :param task: task number
        :param train: training or testing data

        Returns
        -------
        :return: data
        """
        self.get_task_mapping(path)

        if task is None:
            all_lines = {}
            for task_num in self.task_map.keys():
                for file in self.task_map[task_num]:
                    split = file.split("_")
                    # training
                    if split[-1] == "train.txt" and train:
                        all_lines[task_num] = self.read_file(file)
                    # testing
                    elif split[-1] == "test.txt" and not train:
                        all_lines[task_num] = self.read_file(file)
            return all_lines
        else:
            lines = {task: []}
            for file in self.task_map[task]:
                split = file.split("_")
                # training
                if split[-1] == "train.txt" and train:
                    lines[task] = self.read_file(file)
                # testing
                elif split[-1] == "test.txt" and not train:
                    lines[task] = self.read_file(file)
            return lines

    def process_data(self, data) -> dict:
        """
        Process the data.

        Parameters
        ----------
        :param data: data to process

        Returns
        -------
        :return: processed data
        """
        processed_data = {}
        for task in data:
            processed_data[task] = {}
            for id in data[task].keys():
                processed_data[task][id] = {
                    "context": {},
                    "question": {},
                    "answer": {},
                    "supporting_fact": [],
                }
                for line in data[task][id]:
                    cleaned = line.strip()
                    # regex: group 1: line number: \d+\s+
                    # no group: space: \s+
                    # group 2: question: .+?
                    # no group: space: \s+
                    # group 3: answer: \w+
                    # no group: space: \s+
                    # group 4: supporting fact: ((?:\d+\s*)+)
                    question_line_pattern = r"^(\d+)\s+(.+?)\s+(\w+)\s+((?:\d+\s*)+)$"
                    question_match = re.match(question_line_pattern, cleaned)

                    context_line_pattern = r"^(\d+\s+)(.+)$"
                    context_match = re.match(context_line_pattern, cleaned)
                    if question_match:
                        line_num = int(question_match.group(1))
                        processed_data[task][id]["question"][line_num] = (
                            question_match.group(2)
                        )
                        processed_data[task][id]["answer"][line_num] = (
                            question_match.group(3)
                        )
                        supporting_list = [
                            int(x) for x in question_match.group(4).split(" ")
                        ]
                        processed_data[task][id]["supporting_fact"].append(
                            supporting_list
                        )
                    elif context_match:
                        line_num = int(context_match.group(1))
                        processed_data[task][id]["context"][line_num] = (
                            context_match.group(2)
                        )
                    else:
                        print("No match found for line: ", cleaned)

        return processed_data


if __name__ == "__main__":
    dh = DataHandler()
    data = dh.read_data("data/tasks_1-20_v1-2/en", task="qa3", train=True)
    processed = dh.process_data(data)
    print(processed)
