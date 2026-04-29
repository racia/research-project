from __future__ import annotations

import re
from typing import TYPE_CHECKING

from data.utils import expand_cardinal_points, load_scenery
from evaluation.Scenery import Scenery, SentenceScenery
from settings.config import Enumerate, Wrapper

if TYPE_CHECKING:
    from inference.DataLevels import SamplePart

# Scenery fields used for overlap comparison.
# Attribute fields (subj_attributes, obj_attributes) are excluded because
# adjectives recur too freely across unrelated sentences and would inflate
# the distractor count.
_OVERLAP_FIELDS: tuple[str, ...] = (
    "human_subjects",
    "non_human_subjects",
    "direct_objects",
    "indirect_objects",
    "locations",
    "relations",
)


def _normalise_token(tok) -> str:
    """
    Return a lower-cased string from a Scenery token.

    Scenery tokens can be plain strings or tuples (head, *children from
    get_DO_NP); only the head is used for matching so that partial
    noun-phrase overlap counts.

    :param tok: a string or tuple extracted by Scenery
    :return: lower-cased head token as a string
    """
    if isinstance(tok, tuple):
        return str(tok[0]).lower()
    return str(tok).lower()


def _token_set(scenery: SentenceScenery, fields: tuple[str, ...]) -> set[str]:
    """
    Collect all normalised tokens from the selected Scenery fields into a flat set.

    :param scenery: a SentenceScenery instance for one sentence
    :param fields: which SentenceScenery fields to include
    :return: flat set of lower-cased head tokens
    """
    tokens: set[str] = set()
    for field in fields:
        for tok in getattr(scenery, field, []):
            norm = _normalise_token(tok)
            if norm:
                tokens.add(norm)
    return tokens


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

        :param wrapper: wrapper for the data
        :param to_enumerate: enumeration for the data
        """
        self.part_counter: int = 0
        self.sample_counter: int = 0

        self._scenery = Scenery(load_scenery(("base_phrasal_verbs",)) or [])
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

                        question_scenery = self._scenery.extract_from_line(
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

                        context_scenery = self._scenery.extract_from_line(
                            context_match.group(2)
                        )
                        keywords["context"][line_num] = self._entities_from_scenery(
                            context_scenery
                        )

                    else:
                        print("No match found for line: ", cleaned)

        return parts

    def mark_distractors(self, part: "SamplePart") -> None:
        """
        Identify distractor sentences for part and store them in-place as
        part.distractors (list[int]).

        After this call, part.distractors contains the indices of context
        sentences that overlap with the question but are not supporting facts.
        All remaining context sentences are implicitly neutral;
        collect_distractor_attention_record derives that set via set subtraction.

        If the part has no question text, no context, or no extractable
        question tokens, part.distractors is set to [] and the method returns.

        :param part: the SamplePart to annotate; modified in-place
        """
        part.distractors = []

        raw = getattr(part, "raw", None)
        if raw is None:
            return

        # --- question tokens -------------------------------------------------
        question_lines: dict = raw.get("question", {})
        if not question_lines:
            return

        question_text = " ".join(
            v if isinstance(v, str) else " ".join(v) for v in question_lines.values()
        )
        q_scenery = self._scenery.extract_from_line(question_text)
        q_tokens = _token_set(q_scenery, _OVERLAP_FIELDS)

        if not q_tokens:
            return

        # --- context sentences -----------------------------------------------
        context: dict = raw.get("context", {})
        if not context:
            return

        supporting: set[int] = set(part.supporting_sent_inx)
        distractors: list[int] = []

        for sent_idx, sent_text in context.items():
            sent_idx = int(sent_idx)
            if sent_idx in supporting:
                continue

            if isinstance(sent_text, list):
                sent_text = " ".join(sent_text)

            ctx_scenery = self._scenery.extract_from_line(sent_text)
            ctx_tokens = _token_set(ctx_scenery, _OVERLAP_FIELDS)

            if ctx_tokens & q_tokens:
                distractors.append(sent_idx)

        part.distractors = distractors

    def mark_distractors_for_task(self, parts: list["SamplePart"]) -> None:
        """
        Call mark_distractors for every part in the list.

        :param parts: list of SamplePart objects to annotate
        """
        for part in parts:
            self.mark_distractors(part)
