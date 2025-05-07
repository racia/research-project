from __future__ import annotations

import warnings

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from data.utils import load_scenery
from evaluation.Scenery import nlp
from inference.Chat import Chat
from inference.utils import flatten
from interpretability.utils import (
    InterpretabilityResult,
    get_attn_on_target,
    get_indices,
    get_max_attn_ratio,
)


class Interpretability:
    def __init__(self, aggregate_attn: bool = True):
        """
        Initialise Interpretability class
        """
        self.aggregate_attn: bool = aggregate_attn
        # scenery words are only necessary for verbose attention
        self.scenery_words: set[str] = set(map(lambda x: x.lower(), load_scenery()))

        self.tokenizer: PreTrainedTokenizerFast = None

    def get_stop_word_idxs(
        self,
        attn_scores: np.ndarray,
        chat_ids: np.ndarray,
        span_ids: dict = None,
    ) -> list[int]:
        # TODO: update the name and description
        """
        Get indices of stop words in the current task.

        :param chat_ids: current sample part ids (task and model's output)
        :param attn_scores: the attention scores for the current task
        :param span_ids: the sentence spans of the current chat for all types of chunks
        :return: list of indices of stop words in the current task
        """
        assert attn_scores.ndim == 2
        ids_to_remove = []
        task_indices = get_indices(span_ids, "task")
        print("task_indices:", task_indices)
        chat_tokens = self.tokenizer.batch_decode(chat_ids)

        for output_row in attn_scores:
            for task_idx in range(len(output_row)):
                # filter out non-task tokens
                # if task_idx in task_indices:
                #     # ids_to_remove.append(task_idx)
                #     continue
                token = chat_tokens[task_idx].strip().lower()
                print(token)
                for token_ in nlp(token):
                    if token_.lemma_ not in self.scenery_words:
                        ids_to_remove.append(task_idx)
        print(ids_to_remove)
        return ids_to_remove

    @staticmethod
    def get_attention_scores(
        output_tensor: CausalLMOutputWithPast,
        model_output_len: int,
        sent_spans: list[tuple] = None,
        sys_prompt_len: int = 0,
    ) -> np.ndarray:
        """
        Obtains the attention scores from a tensor of attention weights of the current chat.
        The function calculates the attention scores for current task tokens by averaging over layers,
        heads and normalizing over the sum of all token attention scores.

        The following code is an adjusted version of the original implementation from Li et. al 2024
        (Link to paper: https://arxiv.org/abs/2402.18344)

        :param output_tensor: model output tensor
        :param model_output_len: model output length
        :param sent_spans: list of spans of chat sentences without the last model output (if provided, the scores are
        averaged over them)
        :param sys_prompt_len: length of the system prompt (pass only for verbose attention)

        :return: 2D normalized attention scores averaged over layers and heads for the tokens of the current task
        """
        attn_tensor = torch.cat(
            [att.cpu().half() for att in output_tensor["attentions"]], dim=0
        )
        del output_tensor
        # Mean over model layers
        attn_tensor = attn_tensor.mean(dim=0)

        # Takes mean over the attention heads: dimensions, model_output, current task (w/o model output, as it is in
        # y axis)
        attn_tensor = attn_tensor[
            :, -model_output_len:-1, sys_prompt_len:-model_output_len
        ].mean(dim=0)
        # Normalize the attention scores by the sum of all token attention scores
        attn_tensor = attn_tensor / attn_tensor.sum(dim=-1, keepdim=True)
        attn_scores = attn_tensor.float().detach().cpu().numpy()

        if sent_spans:
            # Additionally take mean of attention scores over each task sentence
            attn_scores = np.array(
                [
                    attn_scores[:, start:stop].mean(axis=-1)
                    for (start, stop) in sent_spans
                    if start < stop
                ]
            ).squeeze()
            # Reshape to match expected output format
            if attn_scores.size > 0 and attn_scores.ndim == 2:
                attn_scores_T = attn_scores.transpose(1, 0)
            elif attn_scores.ndim == 1:
                warnings.warn(f"DEBUG: Single row of attention scores:\n{attn_scores}")
                attn_scores_T = attn_scores.reshape(1, -1).T
            elif attn_scores.ndim == 0:
                warnings.warn(f"DEBUG: Empty attention scores:\n{attn_scores}")
                attn_scores_T = attn_scores.reshape(1, -1)
            else:
                warnings.warn(
                    f"DEBUG: Unexpected shape of attention scores:\n{attn_scores}"
                )
                attn_scores_T = attn_scores

            # Normalize the attention scores by special tokens
            # (otherwise the first system prompt sentence gets all the attention)
            attn_scores_T = attn_scores_T / attn_scores_T.sum(axis=0, keepdims=True)

            assert attn_scores_T.shape == (
                attn_scores_T.shape[0],
                len(sent_spans),
            )
            return attn_scores_T

        return attn_scores

    def filter_attn_indices(
        self, attention_scores: np.ndarray, chat_ids: np.ndarray, span_ids: dict = None
    ) -> list:
        """
        Provide indices for scenery words of context and question in each row of the output attention scores.
        Additionally also for message role tokens.

        :param attention_scores: The attention scores of the current chat
        :param chat_ids: current sample part ids (task and model's output)
        :param span_ids: the sentence spans of the current chat for all types of chunks
        :return: according attention_indices
        """
        stop_words_indices = self.get_stop_word_idxs(
            attention_scores, chat_ids, span_ids
        )
        attention_indices = filter(
            lambda x: x not in stop_words_indices, range(attention_scores.shape[1])
        )
        return list(attention_indices)

    def process_attention(
        self,
        output_tensor: CausalLMOutputWithPast,
        chat: Chat,
        chat_ids: torch.Tensor,
        aggregate: bool = True,
    ) -> InterpretabilityResult:
        """
        Process the attention scores and return the interpretability result ready for plotting.

        The following code is an adjusted version of the original implementation from Li et. al 2024
        (Link to paper: https://arxiv.org/abs/2402.18344)

        :param output_tensor: model output tensor for the current chat
        :param chat: the student chat (contains all the messages including the last model output)
        :param chat_ids: the ids of the current chat (including the last model output)
        :param aggregate: if to aggregate the attention scores over the sentences
        :return: InterpretabilityResult object
        """
        # should not include the model output span!
        spans_with_types = chat.get_sentence_spans(remove_last=True)
        sent_spans = list(spans_with_types.keys())
        supp_sent_idx = [
            i for i, span in enumerate(sent_spans) if span in chat.supp_sent_spans
        ]
        # TODO: test verbose attention
        if aggregate:
            # only aggregated sentences, no verbose tokens
            attn_scores = self.get_attention_scores(
                output_tensor=output_tensor,
                model_output_len=len(flatten(chat.messages[-1]["ids"])),
                sent_spans=sent_spans,
            )
            x_tokens = [
                f"* {i} {type_} *" if span in chat.supp_sent_spans else f"{i} {type_}"
                for i, (span, type_) in enumerate(spans_with_types.items(), 1)
            ]
            max_supp_attn_ratio = get_max_attn_ratio(attn_scores, supp_sent_idx)
            attn_on_target = get_attn_on_target(attn_scores, supp_sent_idx)
        else:
            sys_prompt_len = len(flatten(chat.messages[0]["ids"]))
            chat_ids = chat_ids[0][sys_prompt_len + 1 : -1].detach().cpu().numpy()
            attn_scores = self.get_attention_scores(
                output_tensor=output_tensor,
                model_output_len=len(flatten(chat.messages[-1]["ids"])),
                sys_prompt_len=sys_prompt_len,
            )
            attention_indices = self.filter_attn_indices(attn_scores, chat_ids.numpy())
            # Filter attention scores
            attn_scores = attn_scores[:, attention_indices]
            x_tokens = chat.convert_into_datatype(
                datatype="tokens", identify_target=False
            )
            supp_sent_ranges = [
                list(range(*span))
                for span in sent_spans
                if span in chat.supp_sent_spans
            ]
            flat_supp_sent_ranges = flatten(supp_sent_ranges)
            x_tokens = [
                f"* {tok} *" if i in flat_supp_sent_ranges else tok
                for i, tok in enumerate(x_tokens)
                if i in attention_indices
            ]
            max_supp_attn_ratio = get_max_attn_ratio(attn_scores, flat_supp_sent_ranges)
            attn_on_target = get_attn_on_target(attn_scores, flat_supp_sent_ranges)

        if not chat.messages[-1]["tokens"][0]:
            raise ValueError(
                "The last message in the chat does not contain any tokens."
            )

        y_tokens = [
            self.tokenizer.convert_tokens_to_string([token])
            for token in chat.messages[-1]["tokens"][0][:-1]
        ]

        result = InterpretabilityResult(
            attn_scores,
            x_tokens,
            y_tokens,
            max_supp_attn_ratio,
            attn_on_target,
            "aggregated" if aggregate else "verbose",
        )
        return result
