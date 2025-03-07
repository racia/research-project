import os


def get_attn_toks():
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
