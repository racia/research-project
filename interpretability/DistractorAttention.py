from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from inference.DataLevels import SamplePart
from interpretability.utils import (
    _mean_attn_over_indices,
    _sentence_attn_from_interpretability,
)


@dataclass
class DistractorAttentionRecord:
    """
    Holds the distractor and neutral attention values for one SamplePart at one version.
    """

    task_id: int
    sample_id: int
    part_id: int
    version: str
    answer_correct: bool
    attn_distractor: float | None
    attn_supporting: float | None
    attn_neutral: float | None
    n_distractors: int
    n_supporting: int
    n_neutral: int

    def as_dict(self) -> dict:
        """
        Return a plain dictionary representation suitable for CSV serialisation.

        :return: dictionary with all fields; None values are replaced with an empty string
        """
        return {
            "task_id": self.task_id,
            "sample_id": self.sample_id,
            "part_id": self.part_id,
            "version": self.version,
            "answer_correct": self.answer_correct,
            "attn_distractor": (
                "" if self.attn_distractor is None else self.attn_distractor
            ),
            "attn_supporting": (
                "" if self.attn_supporting is None else self.attn_supporting
            ),
            "attn_neutral": "" if self.attn_neutral is None else self.attn_neutral,
            "n_distractors": self.n_distractors,
            "n_supporting": self.n_supporting,
            "n_neutral": self.n_neutral,
        }


class DistractorAttentionStats:
    """
    Accumulates DistractorAttentionRecord objects across parts and exposes
    grouped views consumed by the Plotter methods.
    """

    def __init__(self) -> None:
        """
        Initialise an empty stats container.
        """
        self.records: list[DistractorAttentionRecord] = []

    def add(self, record: DistractorAttentionRecord) -> None:
        """
        Add one record to the collection.

        :param record: the record to add
        """
        self.records.append(record)

    def is_empty(self) -> bool:
        """
        Return True if no records have been added.

        :return: whether the collection is empty
        """
        return len(self.records) == 0

    def as_grouped(self) -> dict[bool, dict[str, list[float]]]:
        """
        Return attention values grouped by answer correctness and sentence role.

        :return: nested dict of the form
                 {True: {"distractor": [...], "supporting": [...], "neutral": [...]},
                  False: {"distractor": [...], "supporting": [...], "neutral": [...]}}
        """
        groups: dict[bool, dict[str, list[float]]] = {
            True: defaultdict(list),
            False: defaultdict(list),
        }
        for r in self.records:
            if r.attn_distractor is not None:
                groups[r.answer_correct]["distractor"].append(r.attn_distractor)
            if r.attn_supporting is not None:
                groups[r.answer_correct]["supporting"].append(r.attn_supporting)
            if r.attn_neutral is not None:
                groups[r.answer_correct]["neutral"].append(r.attn_neutral)
        return groups

    def as_per_task(self) -> dict[int, dict[bool, dict[str, float]]]:
        """
        Return per-task mean attention values grouped by correctness and sentence role.

        :return: nested dict of the form
                 {task_id: {True: {"distractor": mean, "supporting": mean, "neutral": mean}, ...}}
        """
        per_task: dict[int, dict[bool, dict[str, list[float]]]] = defaultdict(
            lambda: {True: defaultdict(list), False: defaultdict(list)}
        )
        for r in self.records:
            if r.attn_distractor is not None:
                per_task[r.task_id][r.answer_correct]["distractor"].append(
                    r.attn_distractor
                )
            if r.attn_supporting is not None:
                per_task[r.task_id][r.answer_correct]["supporting"].append(
                    r.attn_supporting
                )
            if r.attn_neutral is not None:
                per_task[r.task_id][r.answer_correct]["neutral"].append(r.attn_neutral)

        return {
            tid: {
                correct: {role: float(np.mean(vals)) for role, vals in roles.items()}
                for correct, roles in correctness.items()
            }
            for tid, correctness in per_task.items()
        }

    def as_scatter_data(self) -> dict[bool, dict[str, list[float]]]:
        """
        Return only records that have both attn_distractor and attn_neutral.

        :return: nested dict of the form
                 {True: {"distractor": [...], "neutral": [...]},
                  False: {"distractor": [...], "neutral": [...]}}
        """
        out: dict[bool, dict[str, list[float]]] = {
            True: {"distractor": [], "neutral": []},
            False: {"distractor": [], "neutral": []},
        }
        for r in self.records:
            if r.attn_distractor is None or r.attn_neutral is None:
                continue
            out[r.answer_correct]["distractor"].append(r.attn_distractor)
            out[r.answer_correct]["neutral"].append(r.attn_neutral)
        return out

    def as_csv_records(self) -> list[dict]:
        """
        Return all records as plain dicts suitable for saver.save_output.

        :return: list of serialisable dicts
        """
        return [r.as_dict() for r in self.records]

    @property
    def csv_headers(self) -> list[str]:
        """
        Return the CSV column headers matching as_csv_records output.

        :return: list of header strings
        """
        return [
            "task_id",
            "sample_id",
            "part_id",
            "version",
            "answer_correct",
            "attn_distractor",
            "attn_supporting",
            "attn_neutral",
            "n_distractors",
            "n_supporting",
            "n_neutral",
        ]


def collect_distractor_attention_record(
    part: SamplePart,
    answer_correct: bool | int | float,
    version: str,
) -> DistractorAttentionRecord | None:
    """
    Build one DistractorAttentionRecord for a SamplePart at a given version.

    Sentence roles are determined as follows:
      - supporting: indices from part.supporting_sent_inx
      - distractor: indices from part.distractors (set by DataProcessor.mark_distractors)
      - neutral: all context lines that are neither supporting nor distractor

    Returns None when no interpretability data is available for this version
    or when sentence-level attention cannot be extracted.

    :param part: the SamplePart to analyse; must have distractors attribute set
    :param answer_correct: correctness flag for this part at this version
    :param version: "before" or "after"
    :return: a populated DistractorAttentionRecord, or None
    """
    print(
        f"supp={part.supporting_sent_inx} dist={getattr(part, 'distractors', 'MISSING')}"
    )
    result_for_version = None
    for res in part.results:
        if res.version == version:
            result_for_version = res
            break

    if result_for_version is None:
        return None

    context_all = part.raw.get("context_all")
    if context_all is None and part.part_id > 1:
        warnings.warn(
            f"context_all missing for part {part.task_id, part.sample_id, part.part_id}; "
            "previous-part context will not be considered."
        )
    context_all = context_all or part.raw.get("context", {})
    context_line_nums = part.raw.get("context_all_order")
    if context_line_nums is None:
        context_line_nums = list(context_all.keys())
    else:
        missing = set(map(int, context_all.keys())) - set(map(int, context_line_nums))
        if missing:
            warnings.warn(
                f"context_all_order missing {len(missing)} context lines for part "
                f"{part.task_id, part.sample_id, part.part_id}; "
                "sentence-level attention may be misaligned."
            )
    context_line_nums = [int(k) for k in context_line_nums]

    sent_attn = _sentence_attn_from_interpretability(
        result_for_version.interpretability,
        context_line_nums=context_line_nums,
    )
    if sent_attn is None:
        return None

    supporting: set[int] = set(part.supporting_sent_inx)
    distractors: set[int] = set(getattr(part, "distractors", []))
    all_context: set[int] = set(int(k) for k in context_all.keys())
    neutral: set[int] = set(getattr(part, "neutral", [])) or (
        all_context - supporting - distractors
    )

    attn_supporting = _mean_attn_over_indices(sent_attn, list(supporting))
    attn_distractor = _mean_attn_over_indices(sent_attn, list(distractors))
    attn_neutral = _mean_attn_over_indices(sent_attn, list(neutral))

    if len(distractors) == 0 and attn_distractor is None:
        attn_distractor = 0.0

    return DistractorAttentionRecord(
        task_id=part.task_id,
        sample_id=part.sample_id,
        part_id=part.part_id,
        version=version,
        answer_correct=bool(answer_correct),
        attn_supporting=attn_supporting,
        attn_distractor=attn_distractor,
        attn_neutral=attn_neutral,
        n_distractors=len(distractors),
        n_supporting=len(supporting),
        n_neutral=len(neutral),
    )
