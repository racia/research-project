import numpy as np


class InterpretabilityResult:
    def __init__(
        self,
        attn_scores: np.ndarray,
        x_tokens: list[str],
        y_tokens: list[str],
        max_supp_sent: float = None,
    ):
        """
        Interpretability result class
        :param attn_scores: attention scores
        :param x_tokens: tokenized x tokens
        :param y_tokens: tokenized y tokens
        :param max_supp_sent: ratio of max supporting sent
        """
        self.attn_scores: np.ndarray = attn_scores
        self.x_tokens: list[str] = x_tokens
        self.y_tokens: list[str] = y_tokens
        self.max_supp_sent: float = max_supp_sent

        self.result = {
            "attn_scores": self.attn_scores,
            "x_tokens": self.x_tokens,
            "y_tokens": self.y_tokens,
            "max_supp_sent": self.max_supp_sent,
        }

    def __repr__(self) -> str:
        return (
            f"InterpretabilityResult(attn_scores={self.attn_scores.shape}, x_tokens={len(self.x_tokens)}, "
            f"y_tokens={len(self.y_tokens)}, max_supp_sent={self.max_supp_sent})"
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
    attn_scores: np.ndarray,
    supp_sent_spans: list[tuple[int, int]],
    sent_spans: list[tuple[int, int]],
) -> float:
    """
    Returns the ratio of most attended supporting target sentences.

    :param attn_scores: The attention scores
    :param supp_sent_spans: The spans indices of the supporting sentences
    :param sent_spans: The spans indices of the sentences
    :return: Most attended sentence ratio
    """
    max_supp_sent = 0
    supp_sent_idx = [i for i, span in enumerate(sent_spans) if span in supp_sent_spans]
    print("DEBUG supp_sent_idx", supp_sent_idx)

    for output_row in attn_scores:
        # Get index of maximum (mean) attention task token / sentence score
        max_attn_inx = np.argmax(output_row)
        print("max_attn_inx", max_attn_inx)
        if max_attn_inx in supp_sent_idx:
            max_supp_sent += 1

    # Take ratio
    max_supp_sent = max_supp_sent / attn_scores.shape[0]
    return round(max_supp_sent, 4)
