from __future__ import annotations

import numpy as np
import torch

from inference.Chat import Chat, Source
from inference.DataLevels import SamplePart
from interpretability.utils import InterpretabilityResult
from plots.Plotter import Plotter
from settings.Model import Model
from evaluation.Scenery import nlp
from transformers import AutoModelForCausalLM, AutoTokenizer



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

        self.scenery_words: set[str] = scenery_words

    def get_stop_word_idxs(
        self, attn_scores: np.ndarray, chat_ids: torch.LongTensor
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
                token = self.tokenizer.batch_decode(chat_ids)[task_idx]
                token = token.strip()
                for to_lemmatize in nlp(token):
                    lemmatized = to_lemmatize.lemma_
                if lemmatized not in self.scenery_words:
                    stop_words_ids.append(task_idx)
        return stop_words_ids

    @staticmethod
    def get_attention_scores(
        output_tensor: torch.LongTensor,
        model_output_len: int,
    ) -> np.ndarray:
        """
        Obtains the attention scores from a tensor of attention weights of the current chat.
        The function calculates the attention scores for current task tokens by averaging over layers,
        heads and normalizing over the sum of all token attention scores.
        (This code is based on the implementation in https://arxiv.org/abs/2402.18344)

        :param model_output_len: model output length

        :return: 2D normalized attention scores averaged over layers and heads for the tokens of the current task.
        """
        attn_tensor = torch.stack(output_tensor["attentions"], dim=0).squeeze(1)
        # Mean over model layers
        attn_tensor = attn_tensor.mean(dim=0)
        attn_scores = attn_tensor.float().detach().cpu().numpy()

        # Takes mean over the attention heads: dimensions, model_output, current task (w/o system prompt)
        attn_scores = attn_scores[:, -model_output_len:, :-model_output_len].mean(axis=0)
        # Normalize the attention scores by the sum of all token attention scores
        attn_scores = attn_scores / attn_scores.sum(axis=-1, keepdims=True)
        return attn_scores

    def filter_attention_scores(self, attention_scores: np.ndarray, chat_ids: torch.LongTensor) -> tuple[np.ndarray, list]:
        """
        Filter context and question attention scores for scenery words
        by their indices in each row of the output attention scores.
        Removes message role tokens.

        :param attention_scores: The attention scores of the current chat
        :return: filtered attention scores with the according attention_indices
        """
        stop_words_indices = self.get_stop_word_idxs(attention_scores, chat_ids)
        attention_indices = list(
            filter(lambda x: x not in stop_words_indices, range(attention_scores.shape[1]))
        )
        return attention_scores[:, attention_indices], attention_indices

    def get_attention(self, part: SamplePart, chat: Chat) -> InterpretabilityResult:
        """
        1. Defines structural parts of the current chat and gets their input ids and lengths.
        2. Gets the relevant attention scores, filters them.
        3. Constructs x and y tokens and optionally creates heatmaps.
         
        (This code is based on the implementation in https://arxiv.org/abs/2402.18344)

        :param part: part of the sample
        :param chat: Chat history as list of messages
        :return: attention scores, tokenized x and y tokens
        """
        model_output_ids = chat.convert_into_ids(
            chat_part=[chat.messages[-1]],
            max_new_tokens=self.max_new_tokens,
            tokenizer=self.tokenizer,
        )
        model_output_len = len(model_output_ids[0])

        chat_ids = chat.convert_into_ids(
            chat_part=chat.messages[1:],
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

        # Obtain attention scores from model output
        attention_scores = self.get_attention_scores(
            output_tensor=output_tensor,
            model_output_len=model_output_len,
        )

        attention_scores, attention_indices = self.filter_attention_scores(attention_scores, chat_ids)

        # Decode the task tokens
        x_tokens = self.tokenizer.batch_decode(chat_ids[attention_indices])

        # Decode the model output tokens without "assistant" token
        chat_ids = chat_ids[1:]
        y_tokens = self.tokenizer.batch_decode(chat_ids[-model_output_len:])

        if self.save_heatmaps:
            self.plotter.draw_heat(
                x=x_tokens,
                y=y_tokens,
                scores=attention_scores,
                task_id=part.task_id,
                sample_id=part.sample_id,
                part_id=part.part_id,
            )

        torch.cuda.empty_cache()

        return InterpretabilityResult(attention_scores, x_tokens, y_tokens)
