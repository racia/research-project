import os
import re


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

def get_stop_words(tokenizer, input_ids, attn_scores):
    stop_words = []
    attn_tokens = get_attn_toks()
    for cot_row in attn_scores:
        for ind in enumerate(cot_row):
            tok = tokenizer.batch_decode(input_ids)[ind[0]] 
            tok = tok.strip()
            if not (tok.isalpha() or tok in attn_tokens):
                stop_words.append(ind[0])
    return stop_words