from __future__ import annotations

import warnings

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast
import re

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
        chat_tokens: list[str],
    ) -> list[int]:
        # TODO: update the name and description
        """
        Get indices of stop words in the current task.

        :param chat_ids: current sample part ids (task and model's output)
        :param attn_scores: the attention scores for the current task (filtered for task tokens)
        :param span_ids: the sentence spans of the current chat for all types of chunks
        :return: list of indices of stop words in the current task
        """
        assert attn_scores.ndim == 2
        ids_to_remove = []

        for task_idx, _ in enumerate(attn_scores[0, :]): # Iterate over the token indices in the current task span
            try:
                token = chat_tokens[task_idx]
                token_str = self.tokenizer.convert_tokens_to_string([token])
                # print(f"Token str: {token_str}")
                for token_ in nlp(token_str.strip().lower()):
                    if token_.lemma_ not in self.scenery_words:
                        # print(f"Stop word: {token_.lemma_}")
                        ids_to_remove.append(task_idx)
            except IndexError as e:
                print(f"Index of attn_scores ({attn_scores.shape[1]}) exceeds those of chat tokens ({len(chat_tokens)})")
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

        # Takes mean over the attention heads: dimensions, model_output, current task
        # (w/o model output, as it is in y-axis)
        attn_tensor = attn_tensor[
            :, -model_output_len:, sys_prompt_len:-model_output_len # TODO: Check that eot token not included in model output
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
                warnings.warn(
                    f"DEBUG: Single row of attention scores:\n{attn_scores.shape}"
                )
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
            ), f"Unexpected shape of attention scores: {attn_scores_T.shape}, expected (_, {len(sent_spans)})"
            return attn_scores_T

        return attn_scores

    def filter_attn_indices(
        self, attention_scores: np.ndarray, chat_tokens: list, task_spans: list = None, filter_task_spans: bool = True
    ) -> list:
        """
        Provide indices for scenery words of context and question in each row of the output attention scores.
        Additionally also for message role tokens.

        :param attention_scores: The attention scores of the current chat for task
        :param chat_tokens: current sample part tokens
        :param task_spans: the sentence spans of the current chat for all types of chunks
        :param filter_task_spans: whether or not to filter only for task span tokens (excluding model answer tokens)
        :return: according attention_indices
        """
        stop_words_indices = self.get_stop_word_idxs(
            attention_scores, chat_tokens
        )

        if filter_task_spans:
            attention_indices = filter(
                lambda x: x not in stop_words_indices and x in task_spans, range(attention_scores.shape[1])
            )
        else: 
            attention_indices = filter(
                lambda x: x not in stop_words_indices, range(attention_scores.shape[1]) # TODO: Check appearance in task_tokens
            )
        return list(attention_indices)

    def process_attention(
        self,
        output_tensor: CausalLMOutputWithPast,
        chat: Chat,
        chat_ids: torch.Tensor,
    ) -> InterpretabilityResult:
        """
        Process the attention scores and return the interpretability result ready for plotting.

        The following code is an adjusted version of the original implementation from Li et. al 2024
        (Link to paper: https://arxiv.org/abs/2402.18344)

        :param output_tensor: model output tensor for the current chat
        :param chat: the student chat (contains all the messages including the last model output)
        :param chat_ids: the ids of the current chat (including the last model output)
        :return: InterpretabilityResult object
        """
        if not chat.supp_sent_spans:
            raise ValueError("The chat does not contain any supporting sentence spans.")
        # should not include the model output span!
        spans_with_types = chat.get_sentence_spans(remove_last=True)
        sent_spans = list(spans_with_types.keys())
        supp_sent_idx = [
            i for i, span in enumerate(sent_spans) if span in chat.supp_sent_spans
        ]
        model_output_len = len(flatten(chat.messages[-1]["ids"]))
        # TODO: test verbose attention
        if self.aggregate_attn:
            # only aggregated sentences, no verbose tokens
            attn_scores = self.get_attention_scores(
                output_tensor=output_tensor,
                model_output_len=model_output_len,
                sent_spans=sent_spans,
            )
            x_tokens = [
                f"* {i} {type_} *" if span in chat.supp_sent_spans else f"{i} {type_}"
                for i, (span, type_) in enumerate(spans_with_types.items(), 1)
            ]
            max_supp_attn_ratio = get_max_attn_ratio(attn_scores, supp_sent_idx)
            attn_on_target = get_attn_on_target(attn_scores, supp_sent_idx)
        else:
            sys_prompt_len = len(flatten(chat.messages[0]["ids"])) + 4
            chat_ids = chat_ids[0][sys_prompt_len:-model_output_len].detach().cpu().numpy()
            attn_scores = self.get_attention_scores(
                output_tensor=output_tensor,
                model_output_len=model_output_len,
                sys_prompt_len=sys_prompt_len
            )
            task_spans = {spans: type_ for spans, type_ in spans_with_types.items() if type_ in ["cont", "ques", "task"]}
            task_spans_l = list(task_spans.keys())
            try:
                assert sys_prompt_len + len(chat_ids) - 4 == task_spans_l[-1][1]+1 # W/o generation tokens equals last task spans token's index + 1 
            except AssertionError:
                print(f"Unexcepted length of sys prompt: {sys_prompt_len}, chat_ids: {len(chat_ids)} and task tokens: {task_spans_l[-5:]}")
            # print(f"DEBUG: Task spans dict: {task_spans.items()}")
            x_tokens_ = [self.tokenizer.convert_ids_to_tokens([ids])[0] for ids in chat_ids]
            x_tokens = [self.tokenizer.convert_tokens_to_string([tok]) for tok in x_tokens_] 
            
            try:
                # Assert that the x tokens are same length as the task tokens with final model output 
                assert len(x_tokens) == attn_scores.shape[1]
            except AssertionError:
                print(f"Length mismatch between task x_tokens ({len(x_tokens)}) and att_scores ({attn_scores.shape[1]}) (model output: {model_output_len})")
                
            supp_sent_ranges = [    # Get the length of the supporting spans by conforming for sys_prompt_len
                list(range(*np.array(span)-sys_prompt_len))
                for i, span in enumerate(sent_spans, 1)
                if span in chat.supp_sent_spans
            ]
            flat_supp_sent_ranges = flatten(supp_sent_ranges)
            # For optional only task spans tokens filtering 
            task_spans_ranges = [
                list(range(*np.array(span)-sys_prompt_len))
                for span in task_spans_l
            ]
            flat_task_spans_ranges = flatten(task_spans_ranges)
            # In case of model ans spans in x tokens, indices for marking 
            model_ans_spans = chat.get_sentence_spans(span_type="ans", remove_last=True)
            output_span_ranges = [
                list(range(*np.array(span)-sys_prompt_len)) 
                for span, type_ in spans_with_types.items()
                if type_ == "ans"
            ] 
            # print(model_ans_spans, output_span_ranges)
            flat_outp_span_ranges = flatten(output_span_ranges)
            # print(f"Flat outp span ranges ", flat_outp_span_ranges)
            
            # Filter attention scores
            attention_indices = self.filter_attn_indices(attn_scores, x_tokens, task_spans=flat_task_spans_ranges, filter_task_spans=True) # TODO: Possibly try with model ans
            print(f"Supp sent tok range tokens: {np.array(x_tokens)[flat_supp_sent_ranges]}")
            print(f"Model ans range tokens: {np.array(x_tokens)[flat_outp_span_ranges]}")
            
            x_tokens_str_map = {idx: self.tokenizer.convert_tokens_to_string([token]) for idx, token in enumerate(x_tokens)}
            x_tokens = [
                f"* {tok} *" if i in flat_supp_sent_ranges else tok
                for i, tok in enumerate(x_tokens_str_map.values())
                # if i in attention_indices
            ]
            # Additionally mark model answer tokens (if included in x tokens)
            x_tokens = [
                f"_ {tok} _" if i in flat_outp_span_ranges else tok 
                for i, tok in enumerate(x_tokens)
                if i in attention_indices 
            ]
            # print(f"Filtered x tokens ", x_tokens)
            # Compute attention metrics before filtering for correct indexing
            max_supp_attn_ratio = get_max_attn_ratio(attn_scores, flat_supp_sent_ranges) 
            attn_on_target = get_attn_on_target(attn_scores, flat_supp_sent_ranges)
            # print(f"DEBUG: Attn scores shape - before filter: {attn_scores.shape}")
            attn_scores = attn_scores[:, attention_indices] # Filter attention scores to only include those corresponding to the remaining tokens
            # print(f"DEBUG: Attn scores shape - after filter: {attn_scores.shape}")

        if not chat.messages[-1]["tokens"][0]:
            raise ValueError(
                "The last message in the chat does not contain any tokens."
            )

        y_tokens = [
            self.tokenizer.convert_tokens_to_string([token])
            for token in chat.messages[-1]["tokens"][0] # Decoded model output should now exlude <|eot_id|> token
        ]

        result = InterpretabilityResult(
            attn_scores,
            x_tokens,
            y_tokens,
            max_supp_attn_ratio,
            attn_on_target,
            "aggregated" if self.aggregate_attn else "verbose",
        )
        return result
