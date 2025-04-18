import re
import textwrap
from collections import defaultdict
from typing import Any

import en_core_web_sm
from prettytable import PrettyTable
from transformers import PreTrainedTokenizerFast

from evaluation.Evaluator import MetricEvaluator

nlp = en_core_web_sm.load()


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


def generation_tokens(
    tokenizer: PreTrainedTokenizerFast, role: str, eot: bool
) -> list[float]:
    """
    Returns the token id for the role of the message.

    :param tokenizer: tokenizer to use
    :param role: role of the message
    :param eot: whether to add the end of text token
    :return: token id and special tokens
    """
    generation_token = "<|eot_id|>" if eot else "<|begin_of_text|>"

    return tokenizer.convert_tokens_to_ids(
        [generation_token, "<|start_header_id|>", role, "<|end_header_id|>"]
    )


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
                f"{eval_before.max_supp_target.get_mean()} ± {eval_before.max_attn_dist.get_std()}"
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
