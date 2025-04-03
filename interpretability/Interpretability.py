from __future__ import annotations

import warnings

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.Scenery import nlp
from inference.Chat import Chat
from inference.DataLevels import SamplePart
from interpretability.utils import (
    InterpretabilityResult,
    get_supp_tok_idx,
    get_attention_distrib,
)
from plots.Plotter import Plotter
from settings.Model import Model


class Interpretability:
    def __init__(
        self,
        model: Model,
        plotter: Plotter,
        scenery_words: set[str],
        save_heatmaps: bool = False,
    ):
        """
        Interpretability class
        :param model: instance of Model
        :param plotter: instance of Plotter
        :param save_heatmaps: if to create and save heatmaps
        :param scenery_words: set of scenery words
        """
        self.model: AutoModelForCausalLM = model.model
        self.tokenizer: AutoTokenizer = model.tokenizer
        self.max_new_tokens: int = model.max_new_tokens

        self.plotter: Plotter = plotter
        self.save_heatmaps: bool = save_heatmaps

        self.scenery_words: set[str] = set(map(lambda x: x.lower(), scenery_words))

    def get_stop_word_idxs(
        self, attn_scores: np.ndarray, chat_ids: np.ndarray
    ) -> list[int]:
        """
        Get indices of stop words in the current task.

        :param chat_ids: current sample part ids (task and model's output)
        :param attn_scores: the attention scores for the current task
        :return: list of indices of stop words in the current task
        """
        stop_words_ids = []
        for output_row in attn_scores:
            for task_idx in range(len(output_row)):
                token = self.tokenizer.batch_decode(chat_ids)[task_idx].strip().lower()
                for token_ in nlp(token):
                    if token_.lemma_ not in self.scenery_words:
                        stop_words_ids.append(task_idx)
        return stop_words_ids

    @staticmethod
    def get_attention_scores(
        output_tensor: torch.LongTensor,
        model_output_len: int,
        context_sent_spans: list[tuple] = None,
    ) -> np.ndarray:
        """
        Obtains the attention scores from a tensor of attention weights of the current chat.
        The function calculates the attention scores for current task tokens by averaging over layers,
        heads and normalizing over the sum of all token attention scores.

        (The following code is an adjusted version of the original implementation from Li et. al 2024
         (Link to paper: https://arxiv.org/abs/2402.18344))

        :param output_tensor: model output tensor
        :param model_output_len: model output length
        :param context_sent_spans: list of spans of context sentences

        :return: 2D normalized attention scores averaged over layers and heads for the tokens of the current task
        """
        attn_tensor = torch.stack(output_tensor["attentions"], dim=0).squeeze(1)
        # Mean over model layers
        attn_tensor = attn_tensor.mean(dim=0)
        attn_scores = attn_tensor.float().detach().cpu().numpy()

        # Takes mean over the attention heads: dimensions, model_output, current task (w/o system prompt)
        attn_scores = attn_scores[
            :, -model_output_len + 1 :, : -model_output_len + 1
        ].mean(axis=0)

        # Normalize the attention scores by the sum of all token attention scores
        attn_scores = attn_scores / attn_scores.sum(axis=-1, keepdims=True)

        if context_sent_spans:
            # Additionally take mean of attention scores over each task sentence.
            # attn_scores = np.array([attn_scores.T[start:stop] for start, stop in period_indices]).squeeze().mean(
            # axis=-1)
            attn_scores = np.array(
                [
                    attn_scores[:, start:stop].mean(axis=-1)
                    for start, stop in context_sent_spans
                ]
            ).squeeze()
            # Reshape to match expected output format
            attn_scores_T = attn_scores.transpose(1, 0)
            # Normalize the attention scores by the sum of all token attention scores
            attn_scores_T = attn_scores_T / attn_scores_T.sum(axis=0, keepdims=True)

            assert attn_scores_T.shape == (
                attn_scores_T.shape[0],
                len(context_sent_spans),
            )
            return attn_scores_T

        return attn_scores

    def get_tok_indices(
        self, attention_scores: np.ndarray, chat_ids: np.ndarray
    ) -> list:
        """
        Provide indices for scenery words of context and question in each row of the output attention scores.
        Additionally also for message role tokens.

        :param attention_scores: The attention scores of the current chat
        :param chat_ids: current sample part ids (task and model's output)
        :return: according attention_indices
        """
        stop_words_indices = self.get_stop_word_idxs(attention_scores, chat_ids)
        attention_indices = list(
            filter(
                lambda x: x not in stop_words_indices, range(attention_scores.shape[1])
            )
        )
        return attention_indices


    def get_attention(
        self, part: SamplePart, chat: Chat, after: bool = True
    ) -> InterpretabilityResult:
        """
        1. Defines structural parts of the current chat and gets their input ids and lengths.
        2. Gets the relevant attention scores, filters them.
        3. Constructs x and y tokens and optionally creates heatmaps.

        (The following code is an adjusted version of the original implementation from Li et. al 2024 (Link to paper: https://arxiv.org/abs/2402.18344))

        :param part: part of the sample with the output before the setting is applied
        :param chat: Chat history as list of messages
        :param after: if to get attention scores after the setting was applied to the model output or before
        :return: attention scores, tokenized x and y tokens

        """
        chat_messages = chat.messages["student"] if chat.multi_system else chat.messages
        if not chat_messages[-1]["content"]:
            warnings.warn("DEBUG: Interpretability called on empty model output")
            print("DEBUG chat_messages", chat_messages, sep="\n")
            return InterpretabilityResult(np.array(0), [], [], {})

        model_output_ids, _ = chat.convert_into_ids(
            chat_part=[chat_messages[-1]],
            max_new_tokens=self.max_new_tokens,
            tokenizer=self.tokenizer,
        )
        model_output_len = len(model_output_ids[0])

        chat_ids, context_sent_spans = chat.convert_into_ids(
            chat_part=chat_messages[1:],
            max_new_tokens=self.max_new_tokens,
            tokenizer=self.tokenizer,
        )

        # Feed to the model
        output_tensor = self.model(
            input_ids=chat_ids.to(self.model.device),
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )

        chat_ids = chat_ids[0, :].detach().cpu().numpy()

        # Check task length
        overflow = True if len(context_sent_spans) >= 10 else False

        # Obtain attention scores from model output
        attention_scores = self.get_attention_scores(
            output_tensor=output_tensor,
            model_output_len=model_output_len,
            context_sent_spans=context_sent_spans if overflow else None,
        )

        # Get filtering indices
        supp_tok_idx = get_supp_tok_idx(context_sent_spans, part.supporting_sent_inx)

        max_attn_dist = get_attention_distrib(
            attn_scores=attention_scores,
            supp_tok_idx=supp_tok_idx,
            supp_sent_idx=part.supporting_sent_inx if overflow else None,
        )

        if not overflow:
            attention_indices = self.get_tok_indices(attention_scores, chat_ids)
            # Filter attention scores
            attention_scores = attention_scores[:, attention_indices]

            # Decode the task tokens
            x_tokens_ = self.tokenizer.batch_decode(chat_ids)
            x_tokens = []
            for i in range(len(chat_ids)):
                if i in supp_tok_idx:
                    x_tokens.append(f"-- {x_tokens_[i]} --")
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

        # Decode the model output tokens without "user" token
        chat_ids = chat_ids[1:]
        y_tokens = self.tokenizer.batch_decode(chat_ids[-model_output_len + 1 :])

        if self.save_heatmaps:
            self.plotter.draw_heat(
                x=x_tokens,
                x_label="Task Tokens" if not overflow else "Sentence indices",
                y=y_tokens,
                scores=attention_scores,
                task_id=part.task_id,
                sample_id=part.sample_id,
                part_id=part.part_id,
                after=after,
            )

        torch.cuda.empty_cache()

        return InterpretabilityResult(
            attention_scores, x_tokens, y_tokens, max_attn_dist
        )
