import numpy as np


class InterpretabilityResult:
    def __init__(
        self,
        attn_scores: np.ndarray,
        x_tokens: list[str],
        y_tokens: list[str],
        max_attn_dist: dict,
    ):
        """
        Interpretability result class
        :param attn_scores: attention scores
        :param x_tokens: tokenized x tokens
        :param y_tokens: tokenized y tokens
        """
        self.attn_scores: np.ndarray = attn_scores
        self.x_tokens: list[str] = x_tokens
        self.y_tokens: list[str] = y_tokens
        self.max_attn_dist: dict[str, float] = max_attn_dist

        self.result = {
            "attn_scores": self.attn_scores,
            "x_tokens": self.x_tokens,
            "y_tokens": self.y_tokens,
            "max_attn_dist": self.max_attn_dist,
        }

    def __repr__(self) -> str:
        return (
            f"InterpretabilityResult(attn_scores={self.attn_scores.shape}, x_tokens={len(self.x_tokens)}, "
            f"y_tokens={len(self.y_tokens)}, max_attn_dist={self.max_attn_dist})"
        )


def get_supp_tok_idx(
    context_sent_spans: list[tuple[int, int]], supp_sent_idx: list[int]
) -> list[int]:
    """
    Return the indices of the supporting tokens of current chat
    :param context_sent_spans: The indices of sentence spans of current chat (based on chat ids)
    :param supp_sent_idx: the indices of the supporting sentence
    """
    supp_tok_idx = []
    supp_sent_idx = [i - 1 for i in supp_sent_idx]

    for supp_sent_id in supp_sent_idx:
        try:
            supp_tok_range = list(
                range(
                    context_sent_spans[supp_sent_id][0],
                    context_sent_spans[supp_sent_id][1],
                )
            )
            supp_tok_idx.extend(supp_tok_range)
        except IndexError:
            return []
    return supp_tok_idx


def get_attention_distrib(
    attn_scores: np.ndarray, supp_tok_idx, supp_sent_idx: list = None
) -> dict:
    """
    Returns the ratio of most attended tokens for supporting and (non-supporting) other task tokens.
    :param attn_scores: The attention scores
    :param supp_tok_idx: The supporting token indices
    :param supp_sent_idx: The supporting sentence indices
    :return: Most attended token ratio
    """
    supp_name = "sent" if supp_sent_idx else "tok"
    max_supp, max_other = 0, 0

    for output_row in attn_scores:
        # Get index of maximum (mean) attention task token / sentence score
        max_attn_ind = np.argmax(output_row[1:]) # Don't consider high attention "user" token
        supp_range = supp_sent_idx if supp_sent_idx else supp_tok_idx
        # If i is in supporting token indices
        if max_attn_ind in supp_range:
            max_supp += 1
        else:
            max_other += 1

    # Take ratios
    max_supp = max_supp / attn_scores.shape[0]
    max_other = max_other / attn_scores.shape[0]

    return {
        "max_supp_{}".format(supp_name): max_supp,
        "max_other_{}".format(supp_name): max_other,
    }
