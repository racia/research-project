from __future__ import annotations

import re
import warnings
from collections import defaultdict
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from inference.utils import sents_to_ids
from settings.config import Wrapper


def set_device() -> torch.device:
    """
    Sets the device to use for the model.
    If no GPU is available, an error will be raised.

    :return: device to use
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Torch version: ", torch.__version__)
    print("CUDA version: ", torch.version.cuda)
    print("CUDA available: ", torch.cuda.is_available())
    print(f"Device: {device}", flush=True)

    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available. This will be using the CPU.")

    return device


def clean(answer) -> str:
    """
    Cleans the answer if it starts with "not mentioned" but contains additional text.
    This is a common hallucination of the model.
    If the answer starts with "not mentioned" but contains additional text, we will remove the additional text and keep only "not mentioned".
    If the answer does not start with "not mentioned", we will return the answer as is.
    In addition, we will remove the end of text token if it is present in the answer.

    :param answer: the answer to clean
    :return: cleaned answer
    """
    answer = str(answer).removesuffix("<|eot_id|>")
    lower_answer = answer.lower()
    if lower_answer.startswith("not mentioned") and len(answer) > len("not mentioned"):
        addition = answer[len("not mentioned") :].strip()
        answer = lower_answer[: len("not mentioned")]
        warnings.warn(
            f"Hallucinated 'not mentioned' answer contains additional text. Removing: '{addition}', resulting in answer: '{answer}'"
        )
        return answer
    bracket_pattern = re.compile(r"\([\s\S]+?\)$")
    found_bracket = bracket_pattern.search(answer)
    if found_bracket:
        answer = bracket_pattern.sub("", answer).strip()
        warnings.warn(
            f"The answer contains additional text in brackets. Removing: '{found_bracket.group(0)}', resulting in answer: '{answer}'"
        )
    return answer


def not_mentioned_detected(reasoning: str) -> bool:
    """
    Detects if the reasoning contains "not mentioned" which is a common hallucination of the model.
    If "not mentioned" is detected in the reasoning, we will return True.

    :param reasoning: the reasoning to check
    :return: True if "not mentioned" is detected, False otherwise
    """
    phrasings = [
        "not mention",
        "no mention",
        "not specified",
        "not specify",
        "not provide",
        "not enough information",
        # "no information",  # "no information that would contradict..." does not necessarily indicate "not mentioned"
    ]
    return any(phrase in reasoning.lower() for phrase in phrasings)


def parse_output(output: str) -> tuple:
    """
    Parses the output of the model to extract the answer and reasoning.

    :param output: parsed output of the model
    :return: model's answer and reasoning
    """
    answer_pattern = re.compile(r"(?im)^answer:[\s ]*(.+)")
    reasoning_pattern = re.compile(r"(?im)^reasoning:[\s ]*(.+)")

    answer_search = answer_pattern.search(output)
    answer = answer_search[1].strip() if answer_search else ""
    answer = clean(answer)
    # if not answer:
        # print("DEBUG: Answer not found in the output")
    if len(answer) < 2:
        # print(
        #     f"DEBUG: Too short answer: '{answer}', setting to empty string"
        # )
        answer = ""

    reasoning_search = reasoning_pattern.search(output)
    reasoning = reasoning_search[1].strip() if reasoning_search else ""
    if not reasoning or reasoning.lower().startswith("answer:"):
        # print("DEBUG: Reasoning not found in the output")
        reasoning = ""

    if not (answer or reasoning):
        if len(output.split()) <= 3:
            answer = output
        else:
            reasoning = output
        # print(
        #     f"Saving the whole output as the {'answer' if answer else 'reasoning'}:",
        #     output,
        #     sep="\n",
        #     end="\n\n",
        # )
        if not_mentioned_detected(reasoning):
            answer = "not mentioned"
            print(
                "DEBUG: Extracted 'not mentioned' from the reasoning:",
                reasoning,
                sep="\n",
                end="\n\n",
            )

    return answer, reasoning


def encode_wrapper(
    wrapper: Wrapper | dict | str, tokenizer: PreTrainedTokenizerFast
) -> dict[str, dict[str, Any]]:
    """
    Encodes the wrapper into ids and sentence spans. For empty wrapper, there are no values arriving.
    :param wrapper: the wrapper to encode
    :param tokenizer: the tokenizer to use
    :return: tuple of ids and sentence spans
    """
    if not wrapper:
        raise ValueError(
            "Wrapper is not set. Please set the wrapper before calling the model."
        )

    if type(wrapper) is str:
        wrapper = {"wrapper": wrapper}

    wrapper_dict = defaultdict(lambda: defaultdict(dict))
    for key, value in wrapper.items():
        if value:
            no_insert_values = re.split(r" *\{.+?} *", value.strip())
            if len(no_insert_values) > 2:
                raise ValueError(
                    f"The wrapper value '{value}' is not in the correct format. "
                    f"It should be 'wrapper text {{inserted_value}} wrapper text'."
                )
            tokens, ids, sent_spans = sents_to_ids(
                no_insert_values, tokenizer, output_empty=True
            )
            for i, order in enumerate(("before", "after")):
                wrapper_dict[key][order]["content"] = no_insert_values[i]
                wrapper_dict[key][order]["tokens"] = tokens[i]
                wrapper_dict[key][order]["ids"] = ids[i]
                wrapper_dict[key][order]["sent_spans"] = sent_spans[i]
                wrapper_dict[key][order]["spans_with_types"] = (
                    {sent_spans[i]: "wrap"} if sent_spans[i] else {}
                )

    return wrapper_dict
