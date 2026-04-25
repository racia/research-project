from __future__ import annotations

import ast
import warnings
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from inference.DataLevels import SamplePart


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
    attn_neutral: float | None
    n_distractors: int
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
            "attn_neutral": "" if self.attn_neutral is None else self.attn_neutral,
            "n_distractors": self.n_distractors,
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

        Records whose attn_distractor or attn_neutral is None are omitted from
        the corresponding list but do not affect the other role.

        :return: nested dict of the form
                 {True: {"distractor": [...], "neutral": [...]},
                  False: {"distractor": [...], "neutral": [...]}}
        """
        groups: dict[bool, dict[str, list[float]]] = {
            True: defaultdict(list),
            False: defaultdict(list),
        }
        for r in self.records:
            if r.attn_distractor is not None:
                groups[r.answer_correct]["distractor"].append(r.attn_distractor)
            if r.attn_neutral is not None:
                groups[r.answer_correct]["neutral"].append(r.attn_neutral)
        return groups

    def as_per_task(self) -> dict[int, dict[bool, dict[str, float]]]:
        """
        Return per-task mean attention values grouped by correctness and sentence role.

        :return: nested dict of the form
                 {task_id: {True: {"distractor": mean, "neutral": mean}, False: {...}}}
        """
        per_task: dict[int, dict[bool, dict[str, list[float]]]] = defaultdict(
            lambda: {True: defaultdict(list), False: defaultdict(list)}
        )
        for r in self.records:
            if r.attn_distractor is not None:
                per_task[r.task_id][r.answer_correct]["distractor"].append(
                    r.attn_distractor
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
        Return only records that have both attn_distractor and attn_neutral,
        structured for the scatter plot.

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
            "attn_neutral",
            "n_distractors",
            "n_neutral",
        ]


def _sentence_attn_from_interpretability(interpretability) -> dict[int, float] | None:
    """
    Derive a mapping of {bAbI_line_number: mean_attention} from an InterpretabilityResult.

    The attention matrix (output_tokens × input_tokens) is averaged over output tokens
    to yield one weight per input token. Sentence boundaries are found by locating
    bare digit tokens in x_tokens, which correspond to the bAbI line-number prefixes
    embedded in the prompt.

    :param interpretability: an InterpretabilityResult with attn_scores and x_tokens
    :return: dict mapping sentence index to mean attention, or None if extraction fails
    """
    if interpretability is None or interpretability.empty():
        return None

    attn: np.ndarray = interpretability.attn_scores
    x_tokens: list[str] = interpretability.x_tokens

    if attn.size == 0 or not x_tokens:
        return None

    token_weights: np.ndarray = attn.mean(axis=0)

    if len(token_weights) != len(x_tokens):
        warnings.warn(
            f"Token weight length ({len(token_weights)}) does not match x_tokens length "
            f"({len(x_tokens)}); skipping part."
        )
        return None

    sentence_spans: list[tuple[int, int, int]] = []
    current_sent: int | None = None
    start: int = 0

    for i, tok in enumerate(x_tokens):
        if tok.strip().isdigit():
            if current_sent is not None:
                sentence_spans.append((current_sent, start, i))
            current_sent = int(tok.strip())
            start = i + 1

    if current_sent is not None:
        sentence_spans.append((current_sent, start, len(x_tokens)))

    if not sentence_spans:
        return None

    sent_attn: dict[int, float] = {}
    for sent_idx, s, e in sentence_spans:
        span = token_weights[s:e]
        if span.size > 0:
            sent_attn[sent_idx] = float(span.mean())

    return sent_attn if sent_attn else None


def _mean_attn_over_indices(
    sent_attn: dict[int, float], indices: list[int]
) -> float | None:
    """
    Compute the mean attention over a specific set of sentence indices.

    :param sent_attn: mapping of sentence index to mean attention value
    :param indices: sentence indices to average over
    :return: mean attention, or None if no indices overlap with sent_attn
    """
    vals = [sent_attn[i] for i in indices if i in sent_attn]
    return float(np.mean(vals)) if vals else None


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
    result_for_version = None
    for res in part.results:
        if res.version == version:
            result_for_version = res
            break

    if result_for_version is None:
        return None

    sent_attn = _sentence_attn_from_interpretability(
        result_for_version.interpretability
    )
    if sent_attn is None:
        return None

    supporting: set[int] = set(part.supporting_sent_inx)
    distractors: set[int] = set(getattr(part, "distractors", []))
    all_context: set[int] = set(part.raw["context"].keys())
    neutral: set[int] = all_context - supporting - distractors

    return DistractorAttentionRecord(
        task_id=part.task_id,
        sample_id=part.sample_id,
        part_id=part.part_id,
        version=version,
        answer_correct=bool(answer_correct),
        attn_distractor=_mean_attn_over_indices(sent_attn, list(distractors)),
        attn_neutral=_mean_attn_over_indices(sent_attn, list(neutral)),
        n_distractors=len(distractors),
        n_neutral=len(neutral),
    )


def _parse_index_list(val) -> list[int]:
    """
    Parse a list of sentence indices from a stored value, tolerating multiple
    serialisation formats including Python list literals and comma-separated strings.

    :param val: the raw stored value
    :return: list of integer sentence indices
    """
    if isinstance(val, list):
        return [int(x) for x in val]
    if not isinstance(val, str) or not val.strip():
        return []
    try:
        parsed = ast.literal_eval(val)
        return [int(x) for x in parsed]
    except (ValueError, SyntaxError):
        return [int(x) for x in val.split(",") if x.strip().lstrip("-").isdigit()]


def compute_distractor_attention_from_csvs(
    correct_df: pd.DataFrame,
    incorrect_df: pd.DataFrame,
    version: str,
    attn_col_prefix: str = "attn_sentence_",
    distractor_col: str = "distractors",
    supporting_col: str = "supporting_sent_inx",
) -> DistractorAttentionStats:
    """
    Build a DistractorAttentionStats from two pre-filtered DataFrames.

    This is the offline entry point for comparing two separate filtered eval runs,
    e.g. a CSV filtered to correct answers against one filtered to incorrect answers,
    or the two halves of a single CSV split on an answer_correct column.

    Expected columns per row:
      - task_id, sample_id, part_id
      - attn_sentence_1, attn_sentence_2, … (one column per context sentence)
      - distractors: parseable list of sentence indices, e.g. "[3, 7]"
      - supporting_sent_inx: parseable list of supporting fact indices

    :param correct_df: rows where the model answered correctly
    :param incorrect_df: rows where the model answered incorrectly
    :param version: "before" or "after", stored verbatim in each record
    :param attn_col_prefix: prefix for per-sentence attention columns
    :param distractor_col: column name holding distractor sentence indices
    :param supporting_col: column name holding supporting fact sentence indices
    :return: populated DistractorAttentionStats
    """
    stats = DistractorAttentionStats()

    def _row_mean_attn(row: pd.Series, indices: list[int]) -> float | None:
        vals = [
            float(row[f"{attn_col_prefix}{i}"])
            for i in indices
            if f"{attn_col_prefix}{i}" in row.index
            and pd.notna(row[f"{attn_col_prefix}{i}"])
        ]
        return float(np.mean(vals)) if vals else None

    for answer_correct, df in [(True, correct_df), (False, incorrect_df)]:
        for _, row in df.iterrows():
            supporting = set(_parse_index_list(row.get(supporting_col, "")))
            distractors = set(_parse_index_list(row.get(distractor_col, "")))
            attn_cols = [c for c in row.index if c.startswith(attn_col_prefix)]
            all_ctx = {
                int(c.replace(attn_col_prefix, ""))
                for c in attn_cols
                if pd.notna(row[c])
            }
            neutral = all_ctx - supporting - distractors

            stats.add(
                DistractorAttentionRecord(
                    task_id=int(row["task_id"]),
                    sample_id=int(row["sample_id"]),
                    part_id=int(row["part_id"]),
                    version=version,
                    answer_correct=answer_correct,
                    attn_distractor=_row_mean_attn(row, list(distractors)),
                    attn_neutral=_row_mean_attn(row, list(neutral)),
                    n_distractors=len(distractors),
                    n_neutral=len(neutral),
                )
            )

    return stats
