from __future__ import annotations

import re

from data.utils import select_samples
from settings.utils import expand_cardinal_points


class DataProcessor:
    """
    This class preprocesses the data for the models as well as the output of the models.
    """

    def __init__(self):
        """
        Preprocess or postprocess the data.
        """
        self.part_counter = 0

    def process_data(self, data: dict[int, dict], samples_per_task: int = None) -> dict:
        """
        Process the data from a split.

        :param data: data to process
        :param samples_per_task: number of samples to process and return per task
        :return: processed data of type
                 dict[int, dict[str, dict[str, dict[str, str] | dict[str, list[str]] | list[list[int]]]]

                 Example:
                 {
                     task_id: int: {
                        sample_id: str = 0-n:
                        [ # there might be multiple parts for one sample
                             {
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
                                     answers: list[str]
                                 }
                                 "supporting_fact": [
                                     [int], [int, int]
                                 ]
                             }
                         ]
                     }
                 }
        """
        processed_data = {}
        for task in data:
            samples = select_samples(list(data[task].keys()), samples_per_task)
            processed_data[task] = {}
            for sample in samples:

                parts = [
                    {
                        "context": {},
                        "question": {},
                        "answer": {},
                        "supporting_fact": [],
                    }
                ]
                for line in data[task][sample]:
                    cleaned = line.strip()
                    # regex: group 1: line number: \d+\s+
                    # no group: space: \s+
                    # group 2: question: .+?
                    # no group: space: \s+
                    # group 3: answer: \w+(?:,\w+)?     # there might be two answers (see task 8)
                    # no group: space: \s+
                    # group 4: supporting fact: ((?:\d+\s*)+)
                    question_line_pattern = (
                        r"^(\d+)\s+(.+?)\s+(\w+(?:,\w+)?)\s+((?:\d+\s*)+)$"
                    )
                    question_match = re.match(question_line_pattern, cleaned)

                    context_line_pattern = r"^(\d+\s+)(.+)$"
                    context_match = re.match(context_line_pattern, cleaned)

                    if question_match:
                        line_num = int(question_match.group(1))

                        parts[-1]["question"][line_num] = question_match.group(2)
                        # there might be two answers (see task 8)
                        answers = question_match.group(3).lower().split(",")
                        parts[-1]["answer"][line_num] = expand_cardinal_points(answers)

                        supporting_list = [
                            int(x) for x in question_match.group(4).split(" ")
                        ]
                        parts[-1]["supporting_fact"].extend(supporting_list)
                        self.part_counter += 1

                        parts.append(
                            {
                                "context": {},
                                "question": {},
                                "answer": {},
                                "supporting_fact": [],
                            }
                        )

                    elif context_match:
                        line_num = int(context_match.group(1))
                        parts[-1]["context"][line_num] = context_match.group(2)

                    else:
                        print("No match found for line: ", cleaned)

                # Remove the last empty part added after the last question was matched
                parts.pop()
                processed_data[task][sample] = parts

        return processed_data
