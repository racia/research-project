import numpy as np


class InterpretabilityResult:
    def __init__(
        self,
        attn_scores: np.ndarray,
        x_tokens: list[str],
        y_tokens: list[str],
        max_supp_attn: float = None,
        attn_on_target: float = None,
    ):
        """
        Interpretability result class
        :param attn_scores: attention scores
        :param x_tokens: tokenized x tokens
        :param y_tokens: tokenized y tokens
        :param max_supp_attn: ratio of max supporting sent
        :param attn_on_target: average attention on supporting sentences
        """
        self.attn_scores: np.ndarray = attn_scores
        self.x_tokens: list[str] = x_tokens
        self.y_tokens: list[str] = y_tokens
        self.max_supp_attn: float = max_supp_attn
        self.attn_on_target: float = attn_on_target

        self.result = {
            "attn_scores": self.attn_scores,
            "x_tokens": self.x_tokens,
            "y_tokens": self.y_tokens,
            "max_supp_attn": self.max_supp_attn,
            "attn_on_target": self.attn_on_target,
        }

    def __repr__(self) -> str:
        return (
            f"InterpretabilityResult(attn_scores={self.attn_scores.shape}, x_tokens={len(self.x_tokens)}, "
            f"y_tokens={len(self.y_tokens)}, max_supp_attn={self.max_supp_attn}, attn_on_target={self.attn_on_target})"
        )

    def empty(self) -> bool:
        """
        Check if the result is empty.
        :return: True if the result is empty, False otherwise
        """
        print("DEBUG: checking if InterpretabilityResult is empty...")
        print("self.x_tokens", bool(self.x_tokens), self.x_tokens)
        print("self.y_tokens", bool(self.y_tokens), self.y_tokens)
        print("self.attn_scores.shape != ()", self.attn_scores.shape != ())
        if not (self.x_tokens or self.y_tokens or self.attn_scores.shape != ()):
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
    Calculates the percentage of output tokens which maximum attention is on supporting sentences.

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


def get_max_attn_ratio(
    attn_scores: np.ndarray,
    supp_sent_idx: list[int],
) -> float:
    """
    Returns the ratio of most attended supporting target sentences.

    :param attn_scores: The attention scores
    :param supp_sent_idx: The indices of the supporting sentences
    :return: Most attended sentence ratio
    """
    max_attn_inx = np.argmax(attn_scores, axis=1)
    attention_on_supp = np.isin(max_attn_inx, supp_sent_idx)
    max_supp_attn = attention_on_supp.mean()
    # for output_row in attn_scores:
    #     # Get index of maximum (mean) attention task token / sentence score
    #     max_attn_inx = np.argmax(output_row)
    #     if max_attn_inx in supp_sent_idx:
    #         max_supp_attn += 1

    # Take ratio
    # max_supp_attn = max_supp_attn / attn_scores.shape[0]
    return round(float(max_supp_attn), 4)


def get_attn_on_target(
    attn_scores: np.ndarray,
    supp_sent_idx: list[int],
) -> float:
    """
    Calculates the average percentage of attention directed to supporting target sentences.

    :param attn_scores: The attention scores
    :param supp_sent_idx: The indices of the supporting sentences
    :return: Average attention on supporting sentences
    """
    attn_on_supp = attn_scores[:, supp_sent_idx]
    total_attn_per_token = attn_on_supp.sum(axis=1)
    avg_attn_on_supp = total_attn_per_token.mean()
    return round(float(avg_attn_on_supp), 4)
