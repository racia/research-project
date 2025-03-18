from transformers import AutoTokenizer
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


def generation_token(tokenizer: AutoTokenizer, role: str) -> int:
    """
    Returns the token id for the role of the message.

    :param tokenizer: tokenizer to use
    :param role: role of the message
    :return: token id
    """
    if role == "user":
        return tokenizer.convert_tokens_to_ids("user")
    elif role == "assistant":
        return tokenizer.convert_tokens_to_ids("assistant")
    else:
        raise ValueError(f"Unknown role: {role}")
