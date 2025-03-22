from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.Scenery import nlp
from inference.Chat import Chat
from inference.DataLevels import SamplePart
from interpretability.utils import InterpretabilityResult
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
    

    def get_period_idxs(self, chat_ids: torch.LongTensor
        ) -> tuple[int, int]:
        """
        Get tuple of start, stop indices of period from the current task chat ids.

        :param chat_ids: current sample part ids (task and model's output)
        :return: list of indices of periods in the current task
        """
        period_token_id = self.tokenizer.encode(".", add_special_tokens=False)[0]
        period_token_indices = [i for i, tok in enumerate(chat_ids) if tok == period_token_id]
        # Create start, stop tuples of indices
        period_token_indices = [(lambda x:(start, stop))((start,stop)) for start, stop in zip([0]+period_token_indices,period_token_indices)]
        return period_token_indices


    @staticmethod
    def get_attention_scores(
        output_tensor: torch.LongTensor,
        model_output_len: int,
        period_indices: list = None
    ) -> np.ndarray:
        """
        Obtains the attention scores from a tensor of attention weights of the current chat.
        The function calculates the attention scores for current task tokens by averaging over layers,
        heads and normalizing over the sum of all token attention scores.
        (This code is based on the implementation in https://arxiv.org/abs/2402.18344)

        :param model_output_len: model output length

        :return: 2D normalized attention scores averaged over layers and heads for the tokens of the current task
        """
        attn_tensor = torch.stack(output_tensor["attentions"], dim=0).squeeze(1)
        # Mean over model layers
        attn_tensor = attn_tensor.mean(dim=0)
        attn_scores = attn_tensor.float().detach().cpu().numpy()        

        # Takes mean over the attention heads: dimensions, model_output, current task (w/o system prompt)
        attn_scores = attn_scores[:, -model_output_len+1:, :-model_output_len+1].mean(
            axis=0
        )

        # Normalize the attention scores by the sum of all token attention scores
        attn_scores = attn_scores / attn_scores.sum(axis=-1, keepdims=True)

        if period_indices:
            # Additionally take mean of attention scores over each task sentence.        
            #attn_scores = np.array([attn_scores.T[start:stop] for start, stop in period_indices]).squeeze().mean(axis=-1)
            attn_scores = np.array([attn_scores[:,start:stop].mean(axis=-1) for start, stop in period_indices]).squeeze()
            print(attn_scores, attn_scores.shape)
            attn_scores = attn_scores.transpose(1, 0)  # Reshape to match expected output format
            assert attn_scores.shape == (attn_scores.shape[0], len(period_indices))

        return attn_scores

    def filter_attention_scores(
        self, attention_scores: np.ndarray, chat_ids: torch.LongTensor
    ) -> tuple[np.ndarray, list]:
        """
        Filter context and question attention scores for scenery words
        by their indices in each row of the output attention scores.
        Removes message role tokens.

        :param attention_scores: The attention scores of the current chat
        :return: filtered attention scores with the according attention_indices
        """
        stop_words_indices = self.get_stop_word_idxs(attention_scores, chat_ids)
        attention_indices = list(
            filter(
                lambda x: x not in stop_words_indices, range(attention_scores.shape[1])
            )
        )
        return attention_scores[:, attention_indices], attention_indices

    def get_attention(self, part: SamplePart, chat: Chat) -> InterpretabilityResult:
        """
        1. Defines structural parts of the current chat and gets their input ids and lengths.
        2. Gets the relevant attention scores, filters them.
        3. Constructs x and y tokens and optionally creates heatmaps.

        (This code is based on the implementation in https://arxiv.org/abs/2402.18344)

        :param part: part of the sample with the output before the setting is applied
        :param chat: Chat history as list of messages
        :return: attention scores, tokenized x and y tokens
        """

        chat_messages = chat.messages["student"] if chat.multi_system else chat.messages

        model_output_ids = chat.convert_into_ids(
            chat_part=[chat_messages[-1]],
            max_new_tokens=self.max_new_tokens,
            tokenizer=self.tokenizer,
        )
        model_output_len = len(model_output_ids[0])

        chat_ids = chat.convert_into_ids(
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
        period_indices = self.get_period_idxs(chat_ids)
        overflow = True if len(period_indices) >= 10 else False

        # Obtain attention scores from model output
        attention_scores = self.get_attention_scores(
            output_tensor=output_tensor,
            model_output_len=model_output_len,
            period_indices = period_indices if overflow else None
        )

        if not overflow:
            attention_scores, attention_indices = self.filter_attention_scores(
                attention_scores, chat_ids
            )
            # Decode the task tokens
            x_tokens = self.tokenizer.batch_decode(chat_ids[attention_indices])
        
        # Decode the model output tokens without "assistant" token
        chat_ids = chat_ids[1:]
        y_tokens = self.tokenizer.batch_decode(chat_ids[-model_output_len+1:])

        if self.save_heatmaps:
            self.plotter.draw_heat(
                x=x_tokens if not overflow else None,
                y=y_tokens,
                scores=attention_scores,
                task_id=part.task_id,
                sample_id=part.sample_id,
                part_id=part.part_id,
                period_indices=period_indices if overflow else None
            )

        torch.cuda.empty_cache()

        return InterpretabilityResult(attention_scores, x_tokens if not overflow else period_indices, y_tokens)
