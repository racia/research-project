from __future__ import annotations

import warnings

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from evaluation.Scenery import nlp
from inference.Chat import Chat
from inference.DataLevels import SamplePart
from interpretability.utils import (
    InterpretabilityResult,
    get_attn_ratio,
    get_supp_tok_idx,
    get_indices,
)
from plots.Plotter import Plotter


class Interpretability:
    def __init__(
        self,
        scenery_words: set[str],
        plotter: Plotter,
        save_heatmaps: bool = False,
    ):
        """
        Interpretability class
        :param plotter: instance of Plotter
        :param save_heatmaps: if to create and save heatmaps
        :param scenery_words: set of scenery words
        """
        self.scenery_words: set[str] = set(map(lambda x: x.lower(), scenery_words))

        self.plotter: Plotter = plotter
        self.save_heatmaps: bool = save_heatmaps

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
        sys_prompt_len: int = 0,
        sent_spans: list[tuple] = None,
    ) -> np.ndarray:
        """
        Obtains the attention scores from a tensor of attention weights of the current chat.
        The function calculates the attention scores for current task tokens by averaging over layers,
        heads and normalizing over the sum of all token attention scores.

        The following code is an adjusted version of the original implementation from Li et. al 2024
        (Link to paper: https://arxiv.org/abs/2402.18344)

        :param output_tensor: model output tensor
        :param model_output_len: model output length
        :param sys_prompt_len: The system prompt length
        :param sent_spans: list of spans of chat sentences

        :return: 2D normalized attention scores averaged over layers and heads for the tokens of the current task
        """
        attn_tensor = torch.cat(
            [att.cpu().half() for att in output_tensor["attentions"]], dim=0
        )
        del output_tensor
        # Mean over model layers
        attn_tensor = attn_tensor.mean(dim=0)

        # Takes mean over the attention heads: dimensions, model_output, current task (w/o model output, as it is in y axis)
        attn_tensor = attn_tensor[:, -model_output_len:, :-model_output_len].mean(dim=0)

        # Normalize the attention scores by the sum of all token attention scores
        attn_tensor = attn_tensor / attn_tensor.sum(dim=-1, keepdim=True)
        attn_scores = attn_tensor.float().detach().cpu().numpy()

        if sent_spans:
            print("Start averaging attention scores")
            print(attn_scores)
            # Additionally take mean of attention scores over each task sentence.
            attn_scores = np.array(
                [
                    attn_scores[:, start:stop].mean(axis=-1)
                    for (start, stop) in sent_spans
                ]
            ).squeeze()
            # Reshape to match expected output format
            print(f"attn_scores_T shape: {attn_scores.shape}")
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

            # Normalize the attention scores by the sum of all token attention scores
            attn_scores_T = attn_scores_T / attn_scores_T.sum(axis=0, keepdims=True)

            assert attn_scores_T.shape == (
                attn_scores_T.shape[0],
                len(sent_spans),
            )
            print("attn_scores_T:", attn_scores_T)
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

    def calculate_attention(
        self,
        chat_ids: torch.Tensor,
        output_tensor: CausalLMOutputWithPast,
        chat: Chat,
        part: SamplePart,
        after: bool = True,
    ) -> InterpretabilityResult:
        """
        1. Defines structural parts of the current chat and gets their input ids and lengths.
        2. Gets the relevant attention scores, filters them.
        3. Constructs x and y tokens and optionally creates heatmaps.

        The following code is an adjusted version of the original implementation from Li et. al 2024
        (Link to paper: https://arxiv.org/abs/2402.18344)

        :param chat_ids: current sample part ids (task and model's output)
        :param output_tensor: model output tensor
        :param chat: the current student chat
        :param part: part of the sample with the output before the setting is applied
        :param after: if to get attention scores after the setting was applied to the model output or before
        :return: attention scores, tokenized x and y tokens
        """
        sys_prompt_len = len(chat.messages[0]["ids"])
        context_sent_spans = chat.get_sentence_spans(span_type="task")
        model_output_len = len(chat.messages[-1]["ids"])

        chat_ids = chat_ids[0][sys_prompt_len + 1 : -1].detach().cpu().numpy()

        # Check task length
        overflow = True if len(context_sent_spans) >= 10 else False

        # Obtain attention scores from model output
        attention_scores = self.get_attention_scores(
            output_tensor=output_tensor,
            model_output_len=model_output_len,
            sent_spans=context_sent_spans if overflow else None,
            sys_prompt_len=sys_prompt_len,
        )

        # Get filtering indices
        supp_tok_idx = get_supp_tok_idx(context_sent_spans, part.supporting_sent_inx)
        # TODO: fix the attention ratio (outputs 0.0)
        max_attn_dist = get_attn_ratio(
            attn_scores=attention_scores,
            supp_tok_idx=supp_tok_idx,
            supp_sent_idx=part.supporting_sent_inx if overflow else None,
        )

        if not overflow:
            attention_indices = self.filter_attn_indices(attention_scores, chat_ids)
            # Filter attention scores
            attention_scores = attention_scores[:, attention_indices]

            x_tokens = []
            # Decode the task tokens without the system prompt
            x_tokens_ = self.tokenizer.batch_decode(chat_ids)
            torch.cuda.empty_cache()
            for i in range(len(chat_ids)):
                if i in supp_tok_idx:
                    x_tokens.append(f"* {x_tokens_[i]} *")
                else:
                    x_tokens.append(f"{x_tokens_[i]}")

            # Filter tokens
            x_tokens = [tok for i, tok in enumerate(x_tokens) if i in attention_indices]

        else:
            x_tokens = []
            for inx in range(1, len(context_sent_spans) + 1):
                if inx in part.supporting_sent_inx:
                    x_tokens.append(f"* {inx} *")
                else:
                    x_tokens.append(f"{inx}")

        y_tokens = self.tokenizer.batch_decode(chat_ids[-model_output_len + 1 :])
        torch.cuda.empty_cache()

        result = InterpretabilityResult(
            attention_scores, x_tokens, y_tokens, max_attn_dist
        )

        if self.save_heatmaps:
            self.plotter.draw_heat(
                interpretability_result=result,
                x_label="Task Tokens" if not overflow else "Sentence indices",
                task_id=part.task_id,
                sample_id=part.sample_id,
                part_id=part.part_id,
                version="after" if after else "before",
            )
        return result

    def process_attention(
        self,
        output_tensor: CausalLMOutputWithPast,
        chat_ids: torch.Tensor,
        chat: Chat,
        part: SamplePart,
    ) -> InterpretabilityResult:
        """
        Process the attention scores and return the interpretability result ready for plotting.

        The following code is an adjusted version of the original implementation from Li et. al 2024
        (Link to paper: https://arxiv.org/abs/2402.18344)

        :param output_tensor: model output tensor
        :param chat_ids: the ids of the whole chat
        :param chat: the student chat
        :param part: the part of the sample to evaluate
        :return: InterpretabilityResult object
        """
        # TODO: Problems:
        # TODO: there's a weird mean on the first example token
        # TODO: the last tokens of the x axis are longer that the scores and progressively grow
        # TODO: the supporting tokens get padded at wrap (sometimes)
        # system_prompt_len = len(chat.messages[0]["ids"])
        model_output_len = len(chat.messages[-1]["ids"])
        sent_spans = chat.get_sentence_spans()
        print("sent_spans:", sent_spans)
        # should return not all of them, but for the last message
        # TODO: there's a difference between the supporting sentences for the current part and in the current part
        # target_sent_spans = chat.messages[-1]["target_sent_spans"]
        print("part supporting sentences:", part.supporting_sent_inx)
        print("target_sent_spans:", chat.target_sent_spans)
        spans_types = chat.get_sentence_spans(span_type="all")

        # no system prompt but with full tokens, not sentence ids
        # all the spans apart from those for the model output at the end
        attn_scores_aggr = self.get_attention_scores(
            output_tensor=output_tensor,
            model_output_len=model_output_len,
            sent_spans=sent_spans[:-1],
            # sys_prompt_len=system_prompt_len,
        )
        # # includes the system prompt but with aggregated sentences
        # attn_scores_aggr = self.get_attention_scores(
        #     output_tensor=output_tensor,
        #     model_output_len=model_output_len,
        #     sent_spans=None,
        # )

        # TODO: not target sentence spans but general? or no supporting indices just target spans?
        target_indices = get_supp_tok_idx(
            chat.target_sent_spans, part.supporting_sent_inx
        )
        max_attn_ratio_aggr = get_attn_ratio(
            attn_scores=attn_scores_aggr,
            supp_tok_idx=target_indices,
            supp_sent_idx=part.supporting_sent_inx,
        )
        # max_attn_ratio_aggr = get_attn_ratio(
        #     attn_scores=attn_scores_aggr,
        #     supp_tok_idx=target_indices,
        #     supp_sent_idx=part.supporting_sent_inx,
        # )

        # TODO: verbose and filtered x tokens (no system prompt)
        # (already doesn't have the model output)
        chat_ids_aggr = chat_ids[0][:].detach().cpu().numpy()
        # stop_words_indices = self.get_stop_word_idxs(
        #     attn_scores_ver, chat_ids_ver, span_ids
        # )
        # attn_indices = filter(
        #     lambda x: x not in stop_words_indices, range(attn_scores_ver.shape[1])
        # )
        # attn_indices = self.filter_attn_indices(attn_scores_ver, chat_ids_ver, span_ids)
        # Filter attention scores
        # attn_scores_ver = attn_scores_ver[:, attn_indices]

        # Decode the task tokens without the system prompt
        # x_tokens = self.tokenizer.batch_decode(chat_ids_ver)
        # torch.cuda.empty_cache()

        # Pad tokens with asterisks
        # x_tokens_ver = [
        #     f"* {x_tokens[i]} *" if i in target_indices else f"{x_tokens[i]}"
        #     for i in range(len(chat_ids_ver))
        # ]

        # we don't need the last part — the model output
        x_tokens_aggr = [
            f"* {type_} {i} *" if span in chat.target_sent_spans else f"{type_} {i}"
            for i, (span, type_) in enumerate(list(spans_types.items())[:-1], 1)
        ]
        # Filter tokens
        # x_tokens_ver = [
        #     token for i, token in enumerate(x_tokens_ver) if i in attn_indices
        # ]
        # x_tokens_aggr = [
        #     f"* {i} *" if i in part.supporting_sent_inx else f"{i}"
        #     for i in range(1, len(sent_spans) + 1)
        # ]

        y_tokens = self.tokenizer.batch_decode(chat_ids_aggr[-model_output_len + 1 :])
        torch.cuda.empty_cache()

        # TODO save verbose and (?) aggregated attention scores and x tokens (y tokens are always verbose)
        result_aggr = InterpretabilityResult(
            attn_scores_aggr, x_tokens_aggr, y_tokens, max_attn_ratio_aggr
        )
        # result_aggr = InterpretabilityResult(
        #     attn_scores_aggr, x_tokens_aggr, y_tokens, max_attn_ratio_aggr
        # )

        self.plotter.draw_heat(
            interpretability_result=result_aggr,
            x_label="Task Tokens",
            task_id=part.task_id,
            sample_id=part.sample_id,
            part_id=part.part_id,
            # title=f"Attention Map for {part.task_id} - {part.sample_id} - {part.part_id}",
        )
        # self.plotter.draw_heat(
        #     interpretability_result=result_aggr,
        #     x_label="Sentence Indices",
        #     task_id=part.task_id,
        #     sample_id=part.sample_id,
        #     part_id=part.part_id,
        # )
        return result_aggr  # result_aggr
