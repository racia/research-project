import ast
import warnings

import numpy as np

_CONTEXT_TYPE_HINTS: tuple[str, ...] = ("cont", "ctx")


def _parse_index_list(val) -> list[int]:
    """
    Parse a list of sentence indices from a stored value, tolerating multiple
    serialisation formats including Python list literals and comma-separated strings.

    :param val: the raw stored value
    :return: list of integer sentence indices
    """
    if isinstance(val, list):
        return [int(x) for x in val]
    if not isinstance(val, str) or not val.strip():
        return []
    try:
        parsed = ast.literal_eval(val)
        return [int(x) for x in parsed]
    except (ValueError, SyntaxError):
        return [int(x) for x in val.split(",") if x.strip().lstrip("-").isdigit()]


def _parse_aggregated_token(tok: str) -> tuple[int, str] | None:
    """
    Parse one aggregated x_token like "* 3 context *" or "5 question" into
    (chat_sentence_position, type_string).

    Returns None if the token does not match the aggregated label format.

    :param tok: one entry from interpretability.x_tokens
    :return: (chat sentence position, type tag) or None
    """
    if not tok:
        return None
    parts = tok.replace("*", " ").split()
    if len(parts) < 2 or not parts[0].isdigit():
        return None
    # Re-join the tail so multi-word type tags like "sys prompt" survive.
    return int(parts[0]), " ".join(parts[1:]).lower()


def _looks_like_context_type(type_: str) -> bool:
    """
    Heuristic: does this type tag look like it labels a context sentence?

    The actual tag string comes from chat.get_sentence_spans and is not
    fixed across project versions. Rather than hard-code an exhaustive list
    of allowed tags, we accept anything that contains "context" or "ctx".

    :param type_: the type tag part of an aggregated x_token
    :return: True if the tag looks like a context label
    """
    type_low = type_.lower()
    return any(hint in type_low for hint in _CONTEXT_TYPE_HINTS)


def _is_aggregated_x_tokens(x_tokens: list[str]) -> bool:
    """
    Decide whether x_tokens are sentence-level aggregated labels rather than
    individual word/sub-word tokens.

    Aggregated tokens have the form "[*] {digit} {type} [*]". Token-level
    streams contain mostly sub-word strings without a leading digit, so a
    simple majority check on "starts-with-digit-then-word" reliably
    discriminates the two formats — without needing to know the exact set
    of type tags.

    :param x_tokens: the candidate token list
    :return: True if x_tokens look like aggregated sentence labels
    """
    if not x_tokens:
        return False
    n_match = sum(1 for tok in x_tokens if _parse_aggregated_token(tok) is not None)
    return n_match >= max(1, len(x_tokens) // 2)


def _aggregated_sentence_attn(
    token_weights: np.ndarray,
    x_tokens: list[str],
    context_line_nums: list[int] | None,
) -> dict[int, float] | None:
    """
    Build {bAbI_line_number: mean_attention} from aggregated sentence labels.

    In aggregated mode each column of attn_scores corresponds to exactly one
    chat sentence. The chat sentence position embedded in x_tokens is *not*
    the bAbI line number — it counts every chat sentence including system
    prompt sentences, the question, etc. To recover bAbI line numbers we map
    the n-th context-typed x_token to the n-th entry of context_line_nums
    (which holds the part's bAbI line numbers in the order they were emitted).

    If context_line_nums is None we cannot recover bAbI line numbers and
    return None.

    :param token_weights: 1D array of per-sentence attention weights
    :param x_tokens: aggregated sentence labels, same length as token_weights
    :param context_line_nums: bAbI line numbers for the part's context, in order
    :return: {bAbI_line_number: attention} or None if no context entry was found
    """
    if context_line_nums is None:
        return None

    sent_attn: dict[int, float] = {}
    n_context_seen = 0

    for i, tok in enumerate(x_tokens):
        parsed = _parse_aggregated_token(tok)
        if parsed is None:
            continue
        _, type_ = parsed
        if not _looks_like_context_type(type_):
            continue
        if n_context_seen < len(context_line_nums):
            line_num = context_line_nums[n_context_seen]
            sent_attn[line_num] = float(token_weights[i])
        n_context_seen += 1

    if n_context_seen != len(context_line_nums) and n_context_seen > 0:
        warnings.warn(
            f"Number of context-typed x_tokens ({n_context_seen}) does not match "
            f"the number of context line numbers for this part "
            f"({len(context_line_nums)}); some sentences may be misaligned."
        )

    return sent_attn if sent_attn else None


def _verbose_sentence_attn(
    token_weights: np.ndarray,
    x_tokens: list[str],
) -> dict[int, float] | None:
    """
    Build {bAbI_line_number: mean_attention} from token-level x_tokens.

    Sentence boundaries are bare digit tokens (the bAbI line-number prefixes
    embedded in the prompt). The bAbI line number is the digit at the start of
    each sentence; the attention is the mean over the tokens up to (but not
    including) the next digit.

    Tokeniser artefacts such as the SentencePiece "▁" or BPE "Ġ" prefix are
    stripped before checking whether a token is a digit. Tokens wrapped in "*"
    (used to highlight supporting facts) are also unwrapped.

    :param token_weights: 1D array of per-token attention weights
    :param x_tokens: token-level x_tokens, same length as token_weights
    :return: {bAbI_line_number: attention} or None if no sentence was found
    """
    sentence_spans: list[tuple[int, int, int]] = []
    current_sent: int | None = None
    start: int = 0

    for i, tok in enumerate(x_tokens):
        cleaned = tok.replace("*", " ").strip()
        # Strip common subword-prefix markers so e.g. "▁1" / "Ġ1" register as digits.
        cleaned = cleaned.lstrip("▁Ġ ")
        first = cleaned.split()[0] if cleaned else ""
        if first.isdigit():
            if current_sent is not None:
                sentence_spans.append((current_sent, start, i))
            current_sent = int(first)
            start = i + 1

    if current_sent is not None:
        sentence_spans.append((current_sent, start, len(x_tokens)))

    if not sentence_spans:
        return None

    sent_attn: dict[int, float] = {}
    for sent_idx, s, e in sentence_spans:
        span = token_weights[s:e]
        if span.size > 0:
            sent_attn[sent_idx] = float(span.mean())

    return sent_attn if sent_attn else None


def _sentence_attn_from_interpretability(
    interpretability,
    context_line_nums: list[int] | None = None,
) -> dict[int, float] | None:
    """
    Derive a mapping of {bAbI_line_number: mean_attention} from an
    InterpretabilityResult.

    The attention matrix (output_tokens × input_tokens) is averaged over the
    output tokens to yield one weight per input position. The function then
    auto-detects whether the saved x_tokens are sentence-level aggregated
    labels or token-level verbose tokens and dispatches accordingly:

    * Aggregated mode (the default produced by Interpretability with
      aggregate_attn=True): each x_token is one chat sentence labelled
      "[*] {chat_sent_pos} {type} [*]". The n-th "context" entry is mapped to
      context_line_nums[n] to recover the bAbI line number.
    * Verbose mode: x_tokens are individual model tokens with bare digit
      tokens marking bAbI line-number prefixes; sentences are formed between
      consecutive digit tokens.

    :param interpretability: an InterpretabilityResult with attn_scores and x_tokens
    :param context_line_nums: bAbI line numbers for the part's context, in chat
        order — only used in aggregated mode; pass None for verbose mode
    :return: dict mapping bAbI line number to mean attention, or None if
        extraction fails
    """
    if interpretability is None or interpretability.empty():
        return None

    attn: np.ndarray = interpretability.attn_scores
    x_tokens: list[str] = interpretability.x_tokens

    if attn.size == 0 or not x_tokens:
        return None

    token_weights: np.ndarray = attn.mean(axis=0)

    if len(token_weights) != len(x_tokens):
        warnings.warn(
            f"Token weight length ({len(token_weights)}) does not match x_tokens length "
            f"({len(x_tokens)}); skipping part."
        )
        return None

    if _is_aggregated_x_tokens(x_tokens):
        return _aggregated_sentence_attn(token_weights, x_tokens, context_line_nums)

    return _verbose_sentence_attn(token_weights, x_tokens)


def _mean_attn_over_indices(
    sent_attn: dict[int, float], indices: list[int]
) -> float | None:
    vals = [sent_attn[i] for i in indices if i in sent_attn]
    return float(np.mean(vals)) if vals else None


class InterpretabilityResult:
    def __init__(
        self,
        attn_scores: np.ndarray,
        x_tokens: list[str],
        y_tokens: list[str],
        max_supp_attn: float = None,
        attn_on_target: float = None,
        type_: str = None,
    ):
        """
        Interpretability result class
        :param attn_scores: attention scores
        :param x_tokens: tokenized x tokens
        :param y_tokens: tokenized y tokens
        :param max_supp_attn: ratio of max supporting sent
        :param attn_on_target: average attention on supporting sentences
        :param type_: keyword for the result (aggregated or verbose)
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
        self.type_: str = type_

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
        if not (
            self.x_tokens or self.y_tokens or self.attn_scores.shape not in [(), (0,)]
        ):
            return True
        return False


def get_indices(spans_with_types: dict, type_: str):
    """
    Get indices for the spans of the current chat for a desired type of chunk.

    :param spans_with_types: the sentence spans of the current chat for all types of chunks
    :param type_: the type of chunk to get indices for
    """
    if type_ not in ("sys", "ex", "wrap", "task", "ans"):
        raise ValueError(
            "Invalid type. Must be one of 'sys', 'ex', 'wrap', 'task', or 'ans'."
        )
    spans = spans_with_types[type_].keys()
    indices = []
    for span in spans:
        indices.extend(range(span[0], span[1] + 1))
    return indices


# def get_supp_tok_idx(
#     context_sent_spans: list[tuple[int, int]], supp_sent_idx: list[int]
# ) -> list[int]:
#     """
#     Calculates the percentage of output tokens which maximum attention is on supporting sentences.
#
#     :param context_sent_spans: The indices of sentence spans of current chat (based on chat ids)
#     :param supp_sent_idx: the indices of the supporting sentence
#     """
#     supp_tok_idx = []
#     for supp_sent_id in supp_sent_idx:
#         try:
#             supp_tok_range = list(
#                 range(
#                     context_sent_spans[supp_sent_id - 1][0],
#                     context_sent_spans[supp_sent_id - 1][1],
#                 )
#             )
#             supp_tok_idx.extend(supp_tok_range)
#         except IndexError:
#             return []
#     return supp_tok_idx


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
    return round(float(max_supp_attn), 4)


def get_attn_on_target(
    attn_scores: np.ndarray,
    supp_sent_idx: list[int],
) -> float:
    """
    Calculates the average percentage of attention directed to supporting target sentences.
    Both attn scores and indices of the supporting sentences should be for the same values:
    either verbose tokens or aggregated into sentences.

    :param attn_scores: The attention scores
    :param supp_sent_idx: The indices of the supporting sentences
    :return: Average attention on supporting sentences
    """
    attn_on_supp = attn_scores[:, supp_sent_idx]
    total_attn_per_token = attn_on_supp.sum(axis=1)
    avg_attn_on_supp = total_attn_per_token.mean()
    print(f"Average attention on supporting sentences: {avg_attn_on_supp}")
    return round(float(avg_attn_on_supp), 4)
