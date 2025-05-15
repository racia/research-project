from __future__ import annotations

import re


def check_match(
    tokens: list[str], string: str, inx: int = None, intervention: str = None
) -> tuple[list, str]:
    """
    Check if the token list matches the string.

    :param tokens: list of tokens that should be checked for a match
    :param string: the string the tokens are matched against
    :param inx: the index up until which the tokens are approved/index of first error
    :param intervention: the intervention that should be added to the string

    :return: Tuple(list, str): the tokens and the string
    """
    if not inx or inx >= len(tokens):
        inx = len(tokens)

    out_tokens = [token.strip() if token else token for token in tokens][:inx]
    pattern = r"\s*" + r"\s*".join(map(re.escape, out_tokens))

    match = re.match(pattern, string)

    if match:
        out_string = str(match.group(0))
    else:
        out_string = " ".join(out_tokens)

    if intervention:
        out_string += (
            " " + intervention
            if intervention.isalpha() and not (intervention in ["ing", "ed", "s"])
            else intervention
        )
        out_tokens = out_tokens + [intervention]

    print(
        f"Checking match for tokens: {tokens[:inx]} and string: {string}. Result: {match}",
        end="\n\n",
        flush=True,
    )

    return out_tokens, out_string
