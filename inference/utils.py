import re

import en_core_web_sm

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
