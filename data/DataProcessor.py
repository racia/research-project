from __future__ import annotations

import os
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

        try:
            self.keywords_with_origin = {}
            for root, dirs, files in os.walk("data/scenery_words"):
                for file in files:
                    if file.endswith(".txt"):
                        with open(os.path.join(root, file), "r") as f:
                            self.keywords_with_origin[file.split(".")[0]] = list(
                                line.strip().lower() for line in f
                            )
            self.keywords = [
                keyword
                for keywords in self.keywords_with_origin.values()
                for keyword in keywords
            ]
        except Exception as e:
            print(f"Error loading scenery words: {e}")
            self.keywords = {}

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
        :param silver_reasoning: the silver reasoning to addition to the data
        :return: processed data of type
        """
        from_zero = False
        parts = []

        for task_id, task in data.items():
            samples = list(task.items())[:samples_per_task]
            self.sample_counter += len(samples)

            if 0 in task.keys():
                from_zero = True

            for sample_id_, sample in samples:
                sample_id = sample_id_ + 1 if from_zero else sample_id_
                part_id = 1
                raw_part = {
                    "context": {},
                    "question": {},
                    "answer": {},
                    "supporting_facts": [],
                }
                keywords = {"questions": {}, "context": {}}

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
                        naive_tokens = question_match.group(2).split(" ")

                        supporting_list = [
                            int(x) for x in question_match.group(4).split(" ")
                        ]
                        raw_part["supporting_facts"].extend(supporting_list)
                        # there might be two answers (see task 8)
                        answers = question_match.group(3).lower().split(",")

                        reasoning = silver_reasoning.get(
                            (task_id, sample_id, part_id), None
                        )
                        for q_keyword in self.keywords:
                            if any(q_keyword == naive for naive in naive_tokens):
                                if line_num not in keywords["questions"]:
                                    keywords["questions"][line_num] = set()
                                keywords["questions"][line_num].add(q_keyword)

                        part = SamplePart(
                            id_=self.part_counter,
                            task_id=task_id,
                            sample_id=sample_id,
                            part_id=part_id,
                            raw=raw_part,
                            golden_answer=" ".join(expand_cardinal_points(answers)),
                            silver_reasoning=(
                                reasoning["silver_reasoning"] if reasoning else None
                            ),
                            multi_system=multi_system,
                            wrapper=self.wrapper,
                            to_enumerate=self.to_enumerate,
                            keywords=keywords,
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
                        naive_tokens = context_match.group(2).split(" ")

                        for c_keyword in self.keywords:
                            if any(c_keyword == naive for naive in naive_tokens):
                                if line_num not in keywords["context"]:
                                    keywords["context"][line_num] = set()
                                keywords["context"][line_num].add(c_keyword)

                    else:
                        print("No match found for line: ", cleaned)

        parts = self.mark_distractors(parts=parts)
        return parts

    def mark_distractors(self, parts: list[SamplePart]) -> list[SamplePart]:
        """
        For each question, mark the context lines that contain a question keyword, but are not a supporting fact,
        as a distractor.

        :param parts: list[SamplePart], list of SampleParts that should be examined for distractors

        :return: list of SamplePart with distractors added as an attribute
        """
        last_part_id = None
        for ix, curr_part in enumerate(parts):
            consider_prev_parts = curr_part.part_id == last_part_id
            last_part_id = curr_part.part_id

            context_keywords_with_line = dict(curr_part.keywords["context"])

            if consider_prev_parts:
                j = ix - 1
                while j >= 0:
                    prev_part = parts[j]
                    context_keywords_with_line.update(prev_part.keywords["context"])
                    j -= 1

            curr_distractors = set()
            question_keywords_with_line = curr_part.keywords["questions"]
            for line_num, keywords_set in question_keywords_with_line.items():
                for question_keyword in keywords_set:
                    for ctx_line, ctx_keywords in context_keywords_with_line.items():
                        if (
                            any(
                                question_keyword == ctx_keyword
                                for ctx_keyword in ctx_keywords
                            )
                            and ctx_line not in curr_part.supporting_sent_inx
                        ):
                            curr_distractors.add(ctx_line)
            if curr_distractors:
                setattr(curr_part, "distractors", list(curr_distractors))
            else:
                setattr(curr_part, "distractors", [])

            print(
                f"Part ID: {curr_part.part_id}, Supporting facts: {curr_part.supporting_sent_inx}, Distractors: {getattr(curr_part, 'distractors', set())}"
            )
        return parts
