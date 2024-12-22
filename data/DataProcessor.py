from __future__ import annotations

import re


class DataProcessor:
    """
    This class preprocesses the data for the models as well as the output of the models.
    """

    def __init__(self):
        """
        Preprocess or postprocess the data.
        """
        self.question_counter = 0

    def process_data(self, data: dict[str, dict]) -> dict:
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
                    question_line_pattern = (
                        r"^(\d+)\s+(.+?)\s+(\w+(?:,\w+)?)\s+((?:\d+\s*)+)$"
                    )
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
