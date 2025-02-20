import re


def parse_output(output: str) -> dict[str, str]:
    """
    Parses the output of the model to extract the answer and reasoning.

    :param output: parsed output of the model
    :return: dictionary with the model answer and reasoning
    """
    answer_pattern = re.compile(r"(?i)(?<=answer:)[\s ]*.+")
    reasoning_pattern = re.compile(r"(?i)((?<=reasoning)|(?<=reason)):[\s ]*.+")

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

    parsed_output = {"model_answer": answer, "model_reasoning": reasoning}
    return parsed_output
