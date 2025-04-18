from __future__ import annotations

import re
import warnings

import torch

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
    if not answer:
        print("DEBUG: Answer not found in the output")

    reasoning_search = reasoning_pattern.search(output)
    reasoning = reasoning_search[1].strip() if reasoning_search else ""
    if not reasoning:
        print("DEBUG: Reasoning not found in the output")

    if not (answer or reasoning):
        if len(output.split()) <= 3:
            answer = output
        else:
            reasoning = output
        print(
            f"Saving the whole output as the {'answer' if answer else 'reasoning'}:",
            output,
            sep="\n",
            end="\n\n",
        )

    return answer, reasoning
