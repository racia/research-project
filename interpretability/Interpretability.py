from __future__ import annotations

import numpy as np
import torch

from inference.Chat import Chat, Source
from inference.DataLevels import SamplePart
from interpretability.utils import InterpretabilityResult
from plots.Plotter import Plotter
from settings.Model import Model


class Interpretability:
    def __init__(
        self,
        model: Model = None,
        plotter: Plotter = None,
        save_heatmaps: bool = False,
        scenery_words: list[str] = None,
    ):
        """
        Interpretability class
        :param model: instance of Model
        :param plotter: instance of Plotter
        :param save_heatmaps: if to create and save heatmaps
        :param scenery_words: list of scenery words
        """
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.max_new_tokens: int = model.max_new_tokens

        self.plotter: Plotter = plotter
        self.save_heatmaps: bool = save_heatmaps

        self.scenery_words: list[str] = scenery_words

    def get_stop_word_idxs(
        self, part_task_out_ids: torch.LongTensor, attn_scores: np.ndarray
    ) -> list[int]:
        """
        Get indices of stop words in the current task.

        :param part_task_out_ids: current sample part ids (task and model's output)
        :param attn_scores: the attention scores for the current task
        :return: list of indices of stop words in the current task
        """
        stop_words_ids = []
        for output_row in attn_scores:
            for task_inx in enumerate(output_row):
                token = self.tokenizer.batch_decode(part_task_out_ids)[task_inx[0]]
                token = token.strip()
                if token not in self.scenery_words or token in Source.options:
                    stop_words_ids.append(task_inx[0])
        return stop_words_ids

    @staticmethod
    def get_attention_scores(
        output_tensor: torch.LongTensor,
        part_task_len: int,
        part_task_out_len: int,
    ) -> np.ndarray:
        """
        Obtains the attention scores from a tensor of attention weights of current sample part.
        The function calculates the attention scores for current task tokens by averaging over layers,
        heads and normalizing over the sum of all token attention scores.

        @TODO cite (Taken by CoT repo)

        :param part_task_len: question length
        :param part_task_out_len: CoT length

        :return: 2D normalized attention scores averaged over layers and heads for the tokens of the current task.
        """
        attn_tensor = torch.stack(output_tensor["attentions"], dim=0).squeeze(1)
        # Mean over model layers
        attn_tensor = attn_tensor.mean(dim=0)
        attn_scores = attn_tensor.float().detach().cpu().numpy()

        # Takes mean over the attention heads: dimensions, model_output, current task (w/o system prompt)
        attn_scores = attn_scores[
            :, part_task_len:part_task_out_len, :part_task_len
        ].mean(axis=0)
        # Normalize the attention scores by the sum of all token attention scores
        attn_scores = attn_scores / attn_scores.sum(axis=-1, keepdims=True)
        return attn_scores

    def get_attention(self, part: SamplePart, chat: Chat) -> InterpretabilityResult:
        """
        (Taken by CoT repo) Obtains attention scores through output attention weights as scores for CoT and question for each part answer.

        :param part: part of the sample
        :param chat: Chat history as list of messages
        :return: attention scores, tokenized x and y tokens
        """
        # Obtain input_ids and lengths of current part
        part_task_ids = chat.convert_into_ids(
            chat_part=[chat.messages[-2]],
            max_new_tokens=self.max_new_tokens,
            tokenizer=self.tokenizer,
        )
        part_task_len = len(part_task_ids[0])
        print(part_task_ids, part_task_len)

        prev_hist_ids = chat.convert_into_ids(
            chat_part=chat.messages[:-3],  # take everything except last 3
            max_new_tokens=self.max_new_tokens,
            tokenizer=self.tokenizer,
        )
        prev_hist_len = len(prev_hist_ids[0])
        print(prev_hist_ids, prev_hist_len)

        part_task_out_ids = chat.convert_into_ids(
            chat_part=chat.messages[-2:],
            max_new_tokens=self.max_new_tokens,
            tokenizer=self.tokenizer,
        )
        part_task_out_len = len(part_task_out_ids[0])
        print(part_task_out_ids, part_task_out_len)
        # Feed to the model
        output_tensor = self.model(
            input_ids=part_task_out_ids.to(self.model.device),
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )

        # TODO: this now removes all the ids apart from the current part
        part_task_out_ids = part_task_out_ids[0, :].detach().cpu().numpy()

        # Obtain attention scores from model output
        attn_scores = self.get_attention_scores(
            output_tensor=output_tensor,
            part_task_len=part_task_len,
            part_task_out_len=part_task_out_len,
        )
        print(attn_scores.shape)

        # Filter for most import model answer tokens
        stop_words_indices = self.get_stop_word_idxs(part_task_out_ids, attn_scores)
        # Filters out the stop words from the task tokens (context and question)
        # by their indices in each row of the output attention scores
        attn_indices = list(
            filter(lambda x: x not in stop_words_indices, range(attn_scores.shape[1]))
        )
        print("stop", stop_words_indices)
        # print(attn_indices)
        # Shape: (Cot; Y_tokens i.e. indices)
        attn_scores = attn_scores[:, attn_indices]

        # Decode the task tokens
        x_tokens = self.tokenizer.batch_decode(part_task_out_ids[attn_indices])
        print(x_tokens)
        # Decode the model output tokens
        y_tokens = self.tokenizer.batch_decode(
            part_task_out_ids[
                part_task_len:part_task_out_len
            ]  # TODO: can we remove part_task_out_len from here?
        )
        print(y_tokens)
        print(attn_scores.shape)

        if self.save_heatmaps:
            self.plotter.draw_heat(
                x=x_tokens,
                y=y_tokens,
                scores=attn_scores,
                task_id=part.task_id,
                sample_id=part.sample_id,
                part_id=part.part_id,
            )

        torch.cuda.empty_cache()

        return InterpretabilityResult(attn_scores, x_tokens, y_tokens)
