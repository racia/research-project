from __future__ import annotations

import os


def check_or_create_directory(path: str) -> None:
    """
    Check if the directory exists, if not create it.

    :param path: path to the directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_token(token: str) -> str:
    """
    Normalize the token by removing punctuation and converting to lowercase.

    :param token: the token to clean
    :return: the normalized token
    """
    return token.strip("-,.:;!?").lower()


def answer_into_list(answer: str) -> list[str]:
    """
    Convert the answer into a list of words. The word "and" is removed.

    :param answer: the answer
    :return: list of words
    """
    answer_list = [normalize_token(t) for t in answer.split(" ") if t != "and"]
    return answer_list


def two_true_one_pred(true: list, pred: list) -> bool:
    """
    Check if the prediction could be misleadingly correct due to
    true answer containing two equal values.

    :param true: the true answer
    :param pred: the predicted answer
    :return: True if the prediction is misleadingly correct, False otherwise
    """
    if len(true) == 2 and len(pred) != 2 and true[0] == true[1]:
        return True
    return False


def misleading_no(true: set[str], pred: set[str]) -> bool:
    """
    Check if the prediction could misleadingly match as correct due to word boundaries.

    :param true: the true answer
    :param pred: the predicted answer
    :return: True if the prediction is misleadingly correct, False otherwise
    """
    if "none" in true and {"no", "one"}.intersection(pred):
        return True
    if {"no", "one"}.intersection(true) and {"none", "mentioned"}.intersection(pred):
        return True
    if "no" in true and "not" in pred:
        return True
    return False


def normalize_numbers(number_s: int | str | list[int | str]) -> str | list[str]:
    """
    Normalize numbers to a word. For example, "1" -> "one".

    :param number_s: the number to normalize, can be a string, int or a list of such items
    :return: the normalized number
    """
    normalize = {
        "0": "none",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    if type(number_s) == int:
        return normalize[str(number_s)]
    if type(number_s) == str:
        if not number_s.isdigit():
            return number_s
        if number_s == "zero":
            return "none"
        return normalize[number_s]
    elif number_s and type(number_s) == list:
        return [normalize_numbers(number) for number in number_s]
    else:
        raise TypeError(
            f"Expected int, str or list of int or str, got {type(number_s)}"
        )
