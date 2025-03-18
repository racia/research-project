import os
import numpy as np

class InterpretabilityResult:
    def __init__(
        self, attn_scores: np.ndarray, x_tokens: list[str], y_tokens: list[str]
    ):
        """
        Interpretability result class
        :param attn_scores: attention scores
        :param x_tokens: tokenized x tokens
        :param y_tokens: tokenized y tokens
        """
        self.attn_scores = attn_scores
        self.x_tokens = x_tokens
        self.y_tokens = y_tokens

        self.result = {
            "attn_scores": self.attn_scores,
            "x_tokens": self.x_tokens,
            "y_tokens": self.y_tokens,
        }



def get_scenery_words():
    """
    Get scenery words from the scenery_words folder.

    :return: list of scenery words for filtering attention scores
    """
    scenery_words = []
    for entry in os.scandir("interpretability/scenery_words"):
        if entry.is_file and entry.name.endswith(".txt"):
            with open(entry.path, "r", encoding="UTF-8") as f:
                scenery_words.extend(f.read().splitlines())
    return scenery_words
