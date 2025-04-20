from __future__ import annotations

import re
import textwrap
from collections import defaultdict
from typing import Any

import en_core_web_sm
import torch
from prettytable import PrettyTable
from spacy.tokens.span import Span
from transformers import PreTrainedTokenizerFast

from evaluation.Evaluator import MetricEvaluator
from settings.config import Enumerate

nlp = en_core_web_sm.load()


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
    :param to_enumerate: if to add line numbers to the beginning of lines
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


def generation_tokens(tokenizer: PreTrainedTokenizerFast, role: str) -> list[float]:
    """
    Returns the token id for the role of the message.

    :param tokenizer: tokenizer to use
    :param role: role of the message
    :return: token id and special tokens
    """
    tokens = ["<|eot_id|>", "<|start_header_id|>", role, "<|end_header_id|>"]
    return tokenizer.convert_tokens_to_ids(tokens)


def sents_to_ids(
    sentences: list[str | Span], tokenizer: PreTrainedTokenizerFast
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """
    Converts a message into ids using the tokenizer.
    Additionally, it saves the start and end index of each sentence.

    :param sentences: structured message to convert
    :param tokenizer: tokenizer to use
    :return: list of lists of ids that represent sentences and list of sentence spans
    """
    ids = []
    sent_spans = []
    for sentence in sentences:
        print("sentence", sentence)
        if type(sentence) == Span:
            sentence = sentence.text
        # \n\n in source produces empty sentences
        if not sentence or sentence.isspace():
            continue
        tokenized_sentence = tokenizer.encode(
            sentence,
            add_special_tokens=False,
            return_tensors="pt",
        )[0].tolist()
        torch.cuda.empty_cache()
        start = len(ids) + 1
        ids.append(tokenized_sentence)
        end = len(ids)
        sent_spans.append((start, end))

    return ids, sent_spans


def flatten(lst: list[list]) -> list:
    """
    Flattens a list of lists into a single list.

    :param lst: list of lists to flatten
    :return: flattened list
    """
    return [item for sublist in lst for item in sublist]


def upd_span(span: tuple[int, int], offset: int) -> tuple[int, int]:
    """
    Update the span by adding an offset.

    :param span: span to update
    :param offset: offset to add
    :return: updated span
    """
    return span[0] + offset, span[1] + offset


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
    eval_before: MetricEvaluator = None,
    eval_after: MetricEvaluator = None,
    id_: Any = None,
) -> None:
    """
    Print a table comparing metrics before and after a process.

    :param eval_before: MetricEvaluator object before the setting was applied
    :param eval_after: MetricEvaluator object after the setting was applied
    :param id_: ID of the data level
    :return: None
    """
    table = PrettyTable()

    if eval_before and eval_after:
        table.field_names = ["Metric", "Before", "After"]
    elif eval_before:
        table.field_names = ["Metric", "Before"]
    elif eval_after:
        table.field_names = ["Metric", "After"]
    else:
        raise ValueError("At least one MetricEvaluator must be provided.")

    metric_values = defaultdict(dict)

    if eval_before:
        metric_values["Exact-match accuracy"]["Before"] = (
            f"{eval_before.exact_match_accuracy.get_mean()} ± {eval_before.exact_match_accuracy.get_std()}"
            if len(eval_before.exact_match_accuracy) > 1
            else eval_before.exact_match_accuracy.get_mean()
        )
        metric_values["Soft-match accuracy"]["Before"] = (
            f"{eval_before.soft_match_accuracy.get_mean()} ± {eval_before.soft_match_accuracy.get_std()}"
            if len(eval_before.soft_match_accuracy) > 1
            else eval_before.soft_match_accuracy.get_mean()
        )
        if eval_before.max_supp_target:
            metric_values["Max attention distribution"]["Before"] = (
                f"{eval_before.max_supp_target.get_mean()} ± {eval_before.max_supp_target.get_std()}"
                if len(eval_before.max_supp_target) > 1
                else eval_before.max_supp_target.get_mean()
            )

    if eval_after:
        metric_values["Exact-match accuracy"]["After"] = (
            f"{eval_after.exact_match_accuracy.get_mean()} ± {eval_after.exact_match_accuracy.get_std()}"
            if len(eval_after.exact_match_accuracy) > 1
            else eval_after.exact_match_accuracy.get_mean()
        )
        metric_values["Soft-match accuracy"]["After"] = (
            f"{eval_after.soft_match_accuracy.get_mean()} ± {eval_after.soft_match_accuracy.get_std()}"
            if len(eval_after.soft_match_accuracy) > 1
            else eval_after.soft_match_accuracy.get_mean()
        )
        if eval_after.max_supp_target:
            metric_values["Max attention distribution"]["After"] = (
                f"{eval_after.max_supp_target.get_mean()} ± {eval_after.max_supp_target_std.get_std()}"
                if len(eval_after.max_supp_target) > 1
                else eval_after.max_supp_target.get_mean()
            )

    for metric_name, values in metric_values.items():
        row = [metric_name]
        if eval_before:
            row.append(values["Before"])
        if eval_after:
            row.append(values["After"])
        table.add_row(row)

    if id_:
        print(
            f"\nMetrics for {eval_after.level if eval_after else eval_before.level} {id_}:"
        )
    print(table)
