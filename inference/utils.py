from __future__ import annotations

import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import en_core_web_sm
import torch
from prettytable import PrettyTable
from spacy.tokens.span import Span
from transformers import PreTrainedTokenizerFast

from evaluation.Evaluator import MetricEvaluator
from settings.config import Enumerate

nlp = en_core_web_sm.load()


@dataclass
class Source:
    """
    This class handles the roles of the participants in the conversation.
    """

    system: str = "system"
    user: str = "user"
    assistant: str = "assistant"

    options = (system, user, assistant)


def numerate_lines(lines: dict[int, str]) -> list[str]:
    """
    Adds line numbers to the beginning of each line.

    :param lines: lines to numerate
    :return: numerated lines
    """
    return [f"{i}. {line}" for i, line in lines.items()]


def structure_part(
    part: dict[str, dict[int, str]],
    to_enumerate: Enumerate,
) -> tuple[str, str]:
    """
    Structures the lines into a readable format.

    :param part: part of the sample to structure
    :param to_enumerate: if to addition line numbers to the beginning of lines
    :return: structured context and question
    """
    context = []
    question = []
    if part["context"]:
        if to_enumerate.context:
            context.extend(numerate_lines(part["context"]))
        else:
            context.extend(list(part["context"].values()))
    if to_enumerate.question:
        question.extend(numerate_lines(part["question"]))
    else:
        question.extend(list(part["question"].values()))
    return "\n".join(context).strip(), "\n".join(question).strip()


def contains_there(answer) -> bool:
    """
    Check if 'there', 'here' or 'nowhere' is in an answer.

    :param answer: the answer
    :return: bool
    """
    if answer and re.search(r"\b((?:now|t)?here)\b", answer):
        return True
    return False


def contains_verb(answer) -> bool:
    """
    Check if a verb is in the answer.
    """
    answer = nlp(answer)
    if answer and answer[0].tag_.startswith("VB"):
        return True
    return False


def contains_pronouns(answer) -> bool:
    """
    Check if 'he', 'she', 'it', 'they' or 'we' appears in the answer.

    :param answer: the answer
    :return: bool
    """
    if answer and re.search(r"\b(?:he|she|it|her|him|they|them)\b", answer):
        return True
    return False


def contains_not_mentioned(answer) -> bool:
    """
    Check if the model states in doesn't know the answer through phrases like
     - 'not mentioned'
     - 'no information'
     - 'unknown'

    :param answer: the answer
    :return: bool
    """
    if answer and re.search(r"\bmention|information|unknown", answer):
        return True
    return False


def get_generation_token_ids(
    tokenizer: PreTrainedTokenizerFast, role: str, start: bool = False
) -> list[float]:
    """
    Returns the token id for the role of the message.

    :param tokenizer: tokenizer to use
    :param role: role of the message
    :param start: whether to add the start token
    :return: token id and special tokens
    """
    tokens = [
        "<|eot_id|>" if not start else "<|begin_of_text|>",
        "<|start_header_id|>",
        role,
        "<|end_header_id|>",
    ]
    return tokenizer.convert_tokens_to_ids(tokens)


def sents_to_ids(
    sentences: list[str | Span],
    tokenizer: PreTrainedTokenizerFast,
    output_empty: bool = False,
) -> tuple[list[list[str]], list[list[int]], list[tuple]]:
    """
    Converts a message into ids using the tokenizer.
    Additionally, it saves the start and end index of each sentence.

    :param sentences: structured message to convert
    :param tokenizer: tokenizer to use
    :param output_empty: whether to include empty sentences into the output
    :return: tokens and ids that represent sentences, and sentence spans
    """
    flat_ids = []
    tokens, ids, sent_spans = [], [], []
    for sentence in sentences:
        if type(sentence) == Span:
            sentence = sentence.text
        sentence = sentence.strip() + "\n"
        # \n\n in source produces empty sentences
        if not sentence or sentence.isspace():
            if output_empty:
                tokens.append([])
                ids.append([])
                sent_spans.append(())
            continue
        sentence_tokens = tokenizer.tokenize(sentence)
        sentence_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
        torch.cuda.empty_cache()
        tokens.append(sentence_tokens)
        ids.append(sentence_ids)

        start = len(flat_ids)
        flat_ids.extend(sentence_ids)
        end = len(flat_ids)
        sent_spans.append((start, end))

    return tokens, ids, sent_spans


def flatten(lst: list[list] | list) -> list:
    """
    Flattens a list of lists into a single list.

    :param lst: list of lists to flatten
    :return: flattened list
    """
    if not lst:
        return []
    if all(isinstance(i, list) for i in lst):
        return [item for sublist in lst for item in sublist]
    elif not any(isinstance(obj, list) for obj in lst):
        return lst
    raise TypeError(
        "Expected a list of lists or a list of integers, got:",
        *[type(obj) for obj in lst],
    )


def flatten_message(message: dict) -> list[dict]:
    """
    Flattens the message into a list of dictionaries.

    :param message: message to flatten
    :return: flattened message (list of dictionaries)
    """
    flat_message = []
    spans = list(message["spans_with_types"].items())
    for i in range(len(message["ids"])):
        if not message["ids"][i]:
            continue
        span = spans[i][0]
        flat_message.append(
            {
                "content": message["content"][span[0] : span[1]],
                "ids": message["ids"][i],
                "tokens": message["tokens"][i],
                "spans_with_types": (spans[i],),
            }
        )
    return flat_message


def update_span(span: tuple, offset: int) -> tuple[int, int]:
    """
    Update the span by adding an offset.

    :param span: span to update
    :param offset: offset to addition
    :return: updated span
    """
    if len(span) == 0:
        return offset, offset
    if len(span) != 2:
        raise ValueError(
            "Span must be a tuple of two integers representing the start and end indices."
        )
    return offset, span[1] - span[0] + offset


def context_sentences(text: str) -> int:
    """
    Count the number of context-like sentences in the text.

    :param text: text to check
    :return: number of context-like sentences in the text
    """
    context_sent_pattern = re.compile(r"(?i)^\d+\.[a-z\s]+\.")
    matches = context_sent_pattern.findall(text)
    if matches:
        return len(matches)
    return 0


def wrap_text(text, width=40):
    return "\n".join(textwrap.wrap(text, width))


def print_metrics_table(
    evaluators: list[MetricEvaluator],
    id_: Any = None,
) -> None:
    """
    Print a table comparing metrics before and after a process.

    :param evaluators: MetricEvaluator objects with metrics before and after the setting was applied to compare
    :param id_: ID of the data level
    :return: None
    """
    table = PrettyTable()

    if len(evaluators) == 2:
        versions = ["Before", "After"]
        table.field_names = ["Metric", *versions]
    elif len(evaluators) == 1:
        versions = ["Before"]
        table.field_names = ["Metric", *versions]
    else:
        raise ValueError("Only one or two MetricEvaluators can be provided.")

    metric_values = defaultdict(dict)

    for evaluator, version in zip(evaluators, versions):
        metric_values["Exact-match accuracy"][version] = (
            f"{evaluator.exact_match_accuracy.get_mean()} ± {evaluator.exact_match_accuracy.get_std()}"
            if len(evaluator.exact_match_accuracy) > 1
            else evaluator.exact_match_accuracy.get_mean()
        )
        metric_values["Soft-match accuracy"][version] = (
            f"{evaluator.soft_match_accuracy.get_mean()} ± {evaluator.soft_match_accuracy.get_std()}"
            if len(evaluator.soft_match_accuracy) > 1
            else evaluator.soft_match_accuracy.get_mean()
        )
        if evaluator.max_supp_attn:
            metric_values["Max attention ratio"][version] = (
                f"{evaluator.max_supp_attn.get_mean()} ± {evaluator.max_supp_attn.get_std()}"
                if len(evaluator.max_supp_attn) > 1
                else evaluator.max_supp_attn.get_mean()
            )
        if evaluator.attn_on_target:
            metric_values["Attention on Target"][version] = (
                f"{evaluator.attn_on_target.get_mean()} ± {evaluator.attn_on_target.get_std()}"
                if len(evaluator.attn_on_target) > 1
                else evaluator.attn_on_target.get_mean()
            )
        if evaluator.bleu:
            metric_values["BLEU score"][version] = (
                f"{evaluator.bleu.get_mean()} ± {evaluator.bleu.get_std()}"
                if len(evaluator.bleu) > 1
                else evaluator.bleu.get_mean()
            )
        if evaluator.rouge:
            metric_values["ROUGE score"][version] = (
                f"{evaluator.rouge.get_mean()} ± {evaluator.rouge.get_std()}"
                if len(evaluator.rouge) > 1
                else evaluator.rouge.get_mean()
            )
        if evaluator.meteor:
            metric_values["METEOR score"][version] = (
                f"{evaluator.meteor.get_mean()} ± {evaluator.meteor.get_std()}"
                if len(evaluator.meteor) > 1
                else evaluator.meteor.get_mean()
            )
        if evaluator.max_supp_attn_corr:
            metric_values["Max Attn Ratio Correlation"][version] = (
                f"{evaluator.max_supp_attn_corr.get_mean()} ± {evaluator.max_supp_attn_corr.get_std()}"
                if len(evaluator.max_supp_attn_corr) > 1
                else evaluator.max_supp_attn_corr.get_mean()
            )
        if evaluator.attn_on_target_corr:
            metric_values["Attn on Target Correlation"][version] = (
                f"{evaluator.attn_on_target_corr.get_mean()} ± {evaluator.attn_on_target_corr.get_std()}"
                if len(evaluator.attn_on_target_corr) > 1
                else evaluator.attn_on_target_corr.get_mean()
            )

    for metric_name, values in metric_values.items():
        row = [metric_name]
        for version in versions:
            row.append(values.get(version, None))
        table.add_row(row)

    if id_:
        print(f"\nMetrics for {evaluators[0].level} {id_}:")
    print(table)


def type_is_task(target_type: str, type_: str) -> bool:
    """
    Check if the type is task.
    """
    return target_type == "task" and type_ in ["cont", "ques"]


def is_nan(value: Any) -> bool:
    """
    Check if the value is NaN.

    :param value: value to check
    :return: True if the value is NaN, False otherwise
    """
    if value is None:
        return True
    elif isinstance(value, (float, int)):
        return False
    elif isinstance(value, (list, tuple)):
        return any(is_nan(v) for v in value)
    elif isinstance(value, torch.Tensor):
        return torch.isnan(value).any().item()
    elif isinstance(value, str):
        return value.lower() == "nan"
    return False
