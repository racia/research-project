from __future__ import annotations

import re

from data.Scenery import Scenery, SentenceScenery

from data.utils import expand_cardinal_points
from inference.DataLevels import SamplePart
from settings.config import Enumerate, Wrapper


class DataProcessor:
    """
    This class preprocesses the data for the models as well as the output of the models.
    """

    def __init__(
        self,
        scenery: Scenery,
        wrapper: Wrapper = None,
        to_enumerate: Enumerate = None,
    ):
        """
        Preprocess or postprocess the data.

        :param scenery: a Scenery instance used to extract entities from context and question lines
        :param wrapper: wrapper for the data
        :param to_enumerate: enumeration for the data
        """
        self.part_counter: int = 0
        self.sample_counter: int = 0

        self.scenery: Scenery = scenery
        self.wrapper: Wrapper = wrapper
        self.to_enumerate: Enumerate = to_enumerate

    def _entities_from_scenery(self, sentence_scenery: SentenceScenery) -> set[str]:
        """
        Flatten all entity fields of a SentenceScenery into a single set of strings
        for entity-based distractor matching.

        Tuple entities such as noun phrases are joined with a space so they compare
        consistently with plain string entities.

        :param sentence_scenery: the extracted scenery for one sentence
        :return: flat set of entity strings
        """
        entities = set()
        for field_values in sentence_scenery.get().values():
            for item in field_values:
                if isinstance(item, tuple):
                    entities.add(" ".join(item))
                else:
                    entities.add(str(item))
        return entities

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
        :return: processed data as a list of SamplePart objects
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
                        answers = question_match.group(3).lower().split(",")

                        reasoning = silver_reasoning.get(
                            (task_id, sample_id, part_id), None
                        )

                        question_scenery = self.scenery.extract_from_line(
                            question_match.group(2)
                        )
                        keywords["questions"][line_num] = self._entities_from_scenery(
                            question_scenery
                        )

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
                        keywords = {"questions": {}, "context": {}}
                        part_id += 1

                    elif context_match:
                        line_num = int(context_match.group(1))
                        raw_part["context"][line_num] = context_match.group(2)

                        context_scenery = self.scenery.extract_from_line(
                            context_match.group(2)
                        )
                        keywords["context"][line_num] = self._entities_from_scenery(
                            context_scenery
                        )

                    else:
                        print("No match found for line: ", cleaned)

        parts = self.mark_distractors(parts=parts)
        return parts

    def mark_distractors(self, parts: list[SamplePart]) -> list[SamplePart]:
        """
        For each question, mark the context lines that share an entity with the question
        but are not a supporting fact as a distractor.

        Entity overlap is determined by a set intersection between the entities extracted
        from the question line and the entities extracted from each context line. Only
        context lines from the same sample are considered; across parts within the same
        sample, context from earlier parts is also included.

        :param parts: list of SamplePart objects to examine for distractors
        :return: the same list of SampleParts with the distractors attribute set on each
        """
        last_sample_key = None
        for ix, curr_part in enumerate(parts):
            curr_sample_key = (curr_part.task_id, curr_part.sample_id)
            consider_prev_parts = (
                curr_sample_key == last_sample_key and curr_part.part_id > 1
            )
            last_sample_key = curr_sample_key

            context_keywords_with_line = dict(curr_part.keywords["context"])

            if consider_prev_parts:
                j = ix - 1
                while j >= 0:
                    prev_part = parts[j]
                    if (prev_part.task_id, prev_part.sample_id) != curr_sample_key:
                        break
                    context_keywords_with_line.update(prev_part.keywords["context"])
                    j -= 1

            curr_distractors = set()
            for line_num, question_entities in curr_part.keywords["questions"].items():
                for ctx_line, ctx_entities in context_keywords_with_line.items():
                    if (
                        question_entities & ctx_entities
                        and ctx_line not in curr_part.supporting_sent_inx
                    ):
                        curr_distractors.add(ctx_line)

            curr_part.distractors = sorted(curr_distractors)

        return parts
