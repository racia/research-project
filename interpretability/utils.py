import numpy as np


class InterpretabilityResult:
    def __init__(
        self,
        attn_scores: np.ndarray,
        x_tokens: list[str],
        y_tokens: list[str],
        max_supp_target: float = None,
    ):
        """
        Interpretability result class
        :param attn_scores: attention scores
        :param x_tokens: tokenized x tokens
        :param y_tokens: tokenized y tokens
        :param max_supp_target: ratio of max supporting target
        """
        self.attn_scores: np.ndarray = attn_scores
        self.x_tokens: list[str] = x_tokens
        self.y_tokens: list[str] = y_tokens
        self.max_supp_target: float = max_supp_target

        self.result = {
            "attn_scores": self.attn_scores,
            "x_tokens": self.x_tokens,
            "y_tokens": self.y_tokens,
            "max_supp_target": self.max_supp_target,
        }

    def __repr__(self) -> str:
        return (
            f"InterpretabilityResult(attn_scores={self.attn_scores.shape}, x_tokens={len(self.x_tokens)}, "
            f"y_tokens={len(self.y_tokens)}, max_supp_target={self.max_supp_target})"
        )

    def empty(self) -> bool:
        """
        Check if the result is empty.
        :return: True if the result is empty, False otherwise
        """
        if not (
            self.x_tokens
            or self.y_tokens
            or (self.attn_scores and self.attn_scores.size <= 1)
        ):
            return True
        return False


def get_indices(span_ids: dict, type_: str):
    """
    Get indices for the spans of the current chat for a desired type of chunk.

    :param span_ids: the sentence spans of the current chat for all types of chunks
    :param type_: the type of chunk to get indices for
    """
    if type_ not in ("sys", "ex", "wrap", "task", "ans"):
        raise ValueError(
            "Invalid type. Must be one of 'sys', 'ex', 'wrap', 'task', or 'ans'."
        )
    spans = span_ids[type_].keys()
    indices = []
    for span in spans:
        indices.extend(range(span[0], span[1] + 1))
    return indices


def get_supp_tok_idx(
    context_sent_spans: list[tuple[int, int]], supp_sent_idx: list[int]
) -> list[int]:
    """
    Return the indices of the supporting tokens of current chat
    :param context_sent_spans: The indices of sentence spans of current chat (based on chat ids)
    :param supp_sent_idx: the indices of the supporting sentence
    """
    supp_tok_idx = []
    for supp_sent_id in supp_sent_idx:
        try:
            supp_tok_range = list(
                range(
                    context_sent_spans[supp_sent_id - 1][0],
                    context_sent_spans[supp_sent_id - 1][1],
                )
            )
            supp_tok_idx.extend(supp_tok_range)
        except IndexError:
            return []
    return supp_tok_idx


def get_attn_ratio(
    attn_scores: np.ndarray, supp_tok_idx, supp_sent_idx: list = None
) -> float:
    """
    Returns the ratio of most attended supporting target.

    :param attn_scores: The attention scores
    :param supp_tok_idx: The supporting token indices
    :param supp_sent_idx: The supporting sentence indices
    :return: Most attended target ratio
    """
    max_supp_target = 0

    for output_row in attn_scores:
        # Get index of maximum (mean) attention task token / sentence score
        max_attn_ind = np.argmax(
            output_row[1:]
        )  # Don't consider high attention "user" token
        supp_range = supp_sent_idx if supp_sent_idx else supp_tok_idx
        # If i is in supporting token indices
        if max_attn_ind in supp_range:
            max_supp_target += 1

    # Take ratio
    max_supp_target = max_supp_target / attn_scores.shape[0]

    return max_supp_target
