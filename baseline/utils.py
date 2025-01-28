import re
from typing import Union

import torch

from config.baseline_config import Enumerate


def expand_cardinal_points(abbr_news: list[str]) -> list[str]:
    """
    Expands the abbreviations of cardinal points into full words by checking
    if any word as a list item belongs to possible abbreviations.

    :param abbr_news: list of possible abbreviations
    :return: list of words with cardinal points expanded with order preserved
    """
    cardinal_points = {"n": "north", "e": "east", "w": "west", "s": "south"}
    expanded_news = []
    for abbr in abbr_news:
        if abbr in cardinal_points.keys():
            expanded_news.append(cardinal_points[abbr])
        else:
            expanded_news.append(abbr)
    return expanded_news


def numerate_lines(lines: dict[int, str]) -> list[str]:
    """
    Adds line numbers to the beginning of each line.

    :param lines: lines to numerate
    :return: numerated lines
    """
    return [f"{i}. {line}" for i, line in lines.items()]


def structure_part(
    part: dict[str, dict[int, str]],
    to_enumerate: dict[Union[Enumerate.context, Enumerate.question], bool],
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
    return "\n".join(context), "\n".join(question)


def parse_output(output: str) -> dict[str, str]:
    """
    Parses the output of the model to extract the answer and reasoning.

    :param output: parsed output of the model
    """
    answer_pattern = re.compile(r"(?i)(?<=answer:)[\s ]*.+")
    reasoning_pattern = re.compile(r"(?i)(?<=reasoning:)[\s ]*.+")

    answer = answer_pattern.search(output)
    if not answer:
        answer = "None"
        print("DEBUG: Answer not found in the output")
        print("OUTPUT:\n", output, end="\n\n")
    else:
        answer = answer.group(0).strip()

    reasoning = reasoning_pattern.search(output)
    if not reasoning:
        reasoning = "None"
        print("DEBUG: Reasoning not found in the output")
        print("OUTPUT:\n", output, end="\n\n")
    else:
        reasoning = reasoning.group(0).strip()

    parsed_output = {"answer": answer, "reasoning": reasoning}
    return parsed_output


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
        raise Exception("CUDA is not available. This will be using the CPU.")

    return device
