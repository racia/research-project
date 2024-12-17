import re
from typing import List, Dict
from baseline.config.baseline_config import Enumerate


def expand_cardinal_points(abbr_news: List[str]) -> List[str]:
    """
    Expands the abbreviations of cardinal points into full words by checking
    if any word as a list item belongs to possible abbreviations.

    :param abbr_news: list of possible abbreviations
    :return: list of words with cardinal points expanded
    """
    cardinal_points = {
        "n": "north",
        "e": "east",
        "w": "west",
        "s": "south"
    }
    expanded_news = []
    for abbr in abbr_news:
        if abbr in cardinal_points.keys():
            expanded_news.append(cardinal_points[abbr])
        else:
            expanded_news.append(abbr)
    return expanded_news


def sample_into_parts(sample: Dict[str, Dict[int, str]], to_enumerate: dict[Enumerate, bool]) \
        -> List[List[str]]:
    """
    Goes through the joined sample lines in ascending order and
    partitions in after a question was encountered. Context lines
    and questions might be separately numerated or not.

    :param sample: sample data formatted into a dictionary with hierarchical structure
    :param to_enumerate: if to add line numbers to the beginning of lines,
                         'context' sentences and 'question's are considered separately
    :return: sample separated into parts of context lines finished with a question each
    """

    sample_ordered = sorted(list(sample["context"].items()) +
                            list(sample["question"].items()))
    is_question = (lambda sentence: "?" in sentence)
    parts = [[]]

    for line_id, line in sample_ordered:
        # to_enumerate.context and to_enumerate.question should work
        # as they are still part of config
        if not is_question(line) and to_enumerate.context:
            line = f"{line_id}. {line}"
        elif is_question(line) and to_enumerate.question:
            line = f"{line_id}. {line}"

        parts[-1].append(line)
        if is_question(line) and line_id != len(sample_ordered):
            parts.append([])

    return parts


def parse_output(output: str) -> Dict[str, str]:
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
