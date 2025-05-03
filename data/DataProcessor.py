from __future__ import annotations

import re

from data.utils import expand_cardinal_points
from inference.DataLevels import SamplePart
from settings.config import Enumerate, Wrapper


class DataProcessor:
    """
    This class preprocesses the data for the models as well as the output of the models.
    """

    def __init__(
        self,
        wrapper: Wrapper = None,
        to_enumerate: Enumerate = None,
    ):
        """
        Preprocess or postprocess the data.
        """
        self.part_counter: int = 0
        self.sample_counter: int = 0

        self.wrapper: Wrapper = wrapper
        self.to_enumerate: Enumerate = to_enumerate

    def process_data(
        self,
        data: dict[int, dict],
        samples_per_task: int = None,
        multi_system: bool = False,
        silver_reasoning: dict = None,
    ) -> list[SamplePart]:
        """
        Process the data from a split.

        :param data: data to process
        :param samples_per_task: number of samples to process and return per task
        :param multi_system: whether the chat for one sample consists of multiple systems, i.e. a teacher and a student
        :param silver_reasoning: the silver reasoning to add to the data
        :return: processed data of type
        """
        from_zero = False
        parts = []

        for task_id, task in data.items():
            samples = list(task.items())[:samples_per_task]
            self.sample_counter += len(samples)

            if 0 in task.keys():
                from_zero = True

            for sample_id, sample in samples:
                raw_part = {
                    "context": {},
                    "question": {},
                    "answer": {},
                    "supporting_facts": [],
                }
                part_id = 1
                for line in sample:
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
                        self.part_counter += 1
                        line_num = int(question_match.group(1))

                        raw_part["question"][line_num] = question_match.group(2)

                        supporting_list = [
                            int(x) for x in question_match.group(4).split(" ")
                        ]
                        raw_part["supporting_facts"].extend(supporting_list)
                        # there might be two answers (see task 8)
                        answers = question_match.group(3).lower().split(",")

                        reasoning = silver_reasoning.get(
                            (task_id, sample_id, part_id), None
                        )
                        part = SamplePart(
                            id_=self.part_counter,
                            task_id=task_id,
                            sample_id=sample_id + 1 if from_zero else sample_id,
                            part_id=part_id,
                            raw=raw_part,
                            golden_answer=" ".join(expand_cardinal_points(answers)),
                            silver_reasoning=(
                                reasoning["silver_reasoning"] if reasoning else None
                            ),
                            multi_system=multi_system,
                            wrapper=self.wrapper,
                            to_enumerate=self.to_enumerate,
                        )
                        parts.append(part)

                        raw_part = {
                            "context": {},
                            "question": {},
                            "answer": {},
                            "supporting_facts": [],
                        }
                        part_id += 1

                    elif context_match:
                        line_num = int(context_match.group(1))
                        raw_part["context"][line_num] = context_match.group(2)

                    else:
                        print("No match found for line: ", cleaned)

        return parts
