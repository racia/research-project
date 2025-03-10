import os
import re

class Helper:
    def get_attn_toks(self):
        """
        Gets relevant tokens from according TXT files
        :return: List of relevant tokens for scoring attention
        """
        attn_toks = []
        for entry in os.scandir("interpretability"):
            if entry.is_file and "txt" in entry.name:
                with open(entry.path, "r", encoding="UTF-8") as f:
                    attn_toks.append(f.read().splitlines())
        attn_toks = [tok for tok_l in attn_toks for tok in tok_l]
        return attn_toks

    def is_stop_word(self, tok):
        attn_tokens = self.get_attn_toks()
        return not tok.isalpha() or tok not in attn_tokens

