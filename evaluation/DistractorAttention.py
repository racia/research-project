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

    def as_ratio_data(self, eps: float = 1e-8) -> dict[bool, dict[str, list[float]]]:
        """
        Return distractor-to-supporting and distractor-to-neutral attention
        ratios, grouped by answer correctness.

        Ratios are only included for records where both numerator and
        denominator are non-None. A small epsilon is added to denominators
        to prevent division by zero.

        :param eps: small constant added to denominators (default 1e-8)
        :return: nested dict of the form
                 {True:  {"dist_over_supp": [...], "dist_over_neutral": [...]},
                  False: {"dist_over_supp": [...], "dist_over_neutral": [...]}}
        """
        out: dict[bool, dict[str, list[float]]] = {
            True: {"dist_over_supp": [], "dist_over_neutral": []},
            False: {"dist_over_supp": [], "dist_over_neutral": []},
        }
        for r in self.records:
            correct = r.answer_correct
            if r.attn_distractor is not None and r.attn_supporting is not None:
                out[correct]["dist_over_supp"].append(
                    r.attn_distractor / (r.attn_supporting + eps)
                )
            if r.attn_distractor is not None and r.attn_neutral is not None:
                out[correct]["dist_over_neutral"].append(
                    r.attn_distractor / (r.attn_neutral + eps)
                )
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


# Type tags emitted by Interpretability.process_attention in aggregated mode.
# The full set comes from chat.get_sentence_spans; "context" is the only one we
# care about for distractor analysis, but the others let us recognise the format.
_AGGREGATED_TYPES: frozenset[str] = frozenset(
    {
        "sys_prompt",
        "context",
        "question",
        "model_output",
        "answer",
        "reasoning",
        "user",
        "assistant",
        "system",
    }
)


def _parse_aggregated_token(tok: str) -> tuple[int, str] | None:
    """
    Parse one aggregated x_token like "* 3 context *" or "5 question" into
    (chat_sentence_position, type_string).

    Returns None if the token does not match the aggregated label format.

    :param tok: one entry from interpretability.x_tokens
    :return: (chat sentence position, type tag) or None
    """
    if not tok:
        return None
    parts = tok.replace("*", " ").split()
    if len(parts) < 2 or not parts[0].isdigit():
        return None
    return int(parts[0]), parts[1].lower()


def _is_aggregated_x_tokens(x_tokens: list[str]) -> bool:
    """
    Decide whether x_tokens are sentence-level aggregated labels rather than
    individual word/sub-word tokens.

    Aggregated tokens have the form "[*] {digit} {type} [*]", where {type} is
    one of the known chat-chunk type tags. We require the majority of tokens to
    fit that shape to avoid false positives on verbose token streams that
    happen to contain a few digits.

    :param x_tokens: the candidate token list
    :return: True if x_tokens look like aggregated sentence labels
    """
    if not x_tokens:
        return False
    n_match = 0
    for tok in x_tokens:
        parsed = _parse_aggregated_token(tok)
        if parsed is not None and parsed[1] in _AGGREGATED_TYPES:
            n_match += 1
    return n_match >= max(1, len(x_tokens) // 2)


def _aggregated_sentence_attn(
    token_weights: np.ndarray,
    x_tokens: list[str],
    context_line_nums: list[int] | None,
) -> dict[int, float] | None:
    """
    Build {bAbI_line_number: mean_attention} from aggregated sentence labels.

    In aggregated mode each column of attn_scores corresponds to exactly one
    chat sentence. The chat sentence position embedded in x_tokens is *not*
    the bAbI line number — it counts every chat sentence including system
    prompt sentences, the question, etc. To recover bAbI line numbers we map
    the n-th "context" x_token to the n-th entry of context_line_nums (which
    holds the part's bAbI line numbers in the order they were emitted).

    If context_line_nums is None or shorter than the number of context tokens,
    the missing entries are skipped rather than guessed.

    :param token_weights: 1D array of per-sentence attention weights
    :param x_tokens: aggregated sentence labels, same length as token_weights
    :param context_line_nums: bAbI line numbers for the part's context, in order
    :return: {bAbI_line_number: attention} or None if no context entry was found
    """
    sent_attn: dict[int, float] = {}
    n_context_seen = 0

    for i, tok in enumerate(x_tokens):
        parsed = _parse_aggregated_token(tok)
        if parsed is None:
            continue
        _, type_ = parsed
        if type_ != "context":
            continue
        if context_line_nums is not None and n_context_seen < len(context_line_nums):
            line_num = context_line_nums[n_context_seen]
            sent_attn[line_num] = float(token_weights[i])
        n_context_seen += 1

    if (
        context_line_nums is not None
        and n_context_seen != len(context_line_nums)
        and n_context_seen > 0
    ):
        warnings.warn(
            f"Number of 'context' x_tokens ({n_context_seen}) does not match the "
            f"number of context line numbers for this part "
            f"({len(context_line_nums)}); some sentences may be misaligned."
        )

    return sent_attn if sent_attn else None


def _verbose_sentence_attn(
    token_weights: np.ndarray,
    x_tokens: list[str],
) -> dict[int, float] | None:
    """
    Build {bAbI_line_number: mean_attention} from token-level x_tokens.

    Sentence boundaries are bare digit tokens (the bAbI line-number prefixes
    embedded in the prompt). The bAbI line number is the digit at the start of
    each sentence; the attention is the mean over the tokens up to (but not
    including) the next digit.

    Tokeniser artefacts such as the SentencePiece "▁" or BPE "Ġ" prefix are
    stripped before checking whether a token is a digit. Tokens wrapped in "*"
    (used to highlight supporting facts) are also unwrapped.

    :param token_weights: 1D array of per-token attention weights
    :param x_tokens: token-level x_tokens, same length as token_weights
    :return: {bAbI_line_number: attention} or None if no sentence was found
    """
    sentence_spans: list[tuple[int, int, int]] = []
    current_sent: int | None = None
    start: int = 0

    for i, tok in enumerate(x_tokens):
        cleaned = tok.replace("*", " ").strip()
        # Strip common subword-prefix markers so e.g. "▁1" / "Ġ1" register as digits.
        cleaned = cleaned.lstrip("▁Ġ ")
        first = cleaned.split()[0] if cleaned else ""
        if first.isdigit():
            if current_sent is not None:
                sentence_spans.append((current_sent, start, i))
            current_sent = int(first)
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


def _sentence_attn_from_interpretability(
    interpretability,
    context_line_nums: list[int] | None = None,
) -> dict[int, float] | None:
    """
    Derive a mapping of {bAbI_line_number: mean_attention} from an
    InterpretabilityResult.

    The attention matrix (output_tokens × input_tokens) is averaged over the
    output tokens to yield one weight per input position. The function then
    auto-detects whether the saved x_tokens are sentence-level aggregated
    labels or token-level verbose tokens and dispatches accordingly:

    * Aggregated mode (the default produced by Interpretability with
      aggregate_attn=True): each x_token is one chat sentence labelled
      "[*] {chat_sent_pos} {type} [*]". The n-th "context" entry is mapped to
      context_line_nums[n] to recover the bAbI line number.
    * Verbose mode: x_tokens are individual model tokens with bare digit
      tokens marking bAbI line-number prefixes; sentences are formed between
      consecutive digit tokens.

    :param interpretability: an InterpretabilityResult with attn_scores and x_tokens
    :param context_line_nums: bAbI line numbers for the part's context, in chat
        order — only used in aggregated mode; pass None for verbose mode
    :return: dict mapping bAbI line number to mean attention, or None if
        extraction fails
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

    if _is_aggregated_x_tokens(x_tokens):
        return _aggregated_sentence_attn(token_weights, x_tokens, context_line_nums)

    return _verbose_sentence_attn(token_weights, x_tokens)


def _mean_attn_over_indices(
    sent_attn: dict[int, float], indices: list[int]
) -> float | None:
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

    # Pass the part's bAbI context line numbers so aggregated-mode x_tokens can
    # be mapped back from chat-sentence position to bAbI line number. Sorted to
    # match the order context sentences are emitted into the prompt.
    context_line_nums = sorted(int(k) for k in part.raw.get("context", {}).keys())
    sent_attn = _sentence_attn_from_interpretability(
        result_for_version.interpretability,
        context_line_nums=context_line_nums,
    )
    if sent_attn is None:
        return None

    supporting: set[int] = set(part.supporting_sent_inx)
    distractors: set[int] = set(getattr(part, "distractors", []))
    all_context: set[int] = set(part.raw["context"].keys())
    neutral: set[int] = all_context - supporting - distractors

    attn_supporting = _mean_attn_over_indices(sent_attn, list(supporting))
    attn_distractor = _mean_attn_over_indices(sent_attn, list(distractors))
    attn_neutral = _mean_attn_over_indices(sent_attn, list(neutral))

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
                    attn_supporting=_row_mean_attn(row, list(supporting)),
                    attn_neutral=_row_mean_attn(row, list(neutral)),
                    n_distractors=len(distractors),
                    n_supporting=len(supporting),
                    n_neutral=len(neutral),
                )
            )

    return stats
