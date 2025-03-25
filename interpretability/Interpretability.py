<<<<<<< HEAD

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch

from plots.Plotter import Plotter


@dataclass
class Interpretability:
    
    switch: bool
    path: str
    sample: bool
    part: bool
    
    def __init__(self, model, tokenizer, task_id):
        """
        """
        self.model = model
        self.tokenizer = tokenizer
        self.task_id = task_id

    def check_tok(self, dist_toks, tok):
        return tok in dist_toks
      
    def build_chat_input(self, model, tokenizer, messages: List[dict], max_new_tokens: int=0):
        def _parse_messages(messages, split_role="user"):
            system, rounds = "", []
            round = []
            for i, message in enumerate(messages):
                if message["role"] == "system":
                    assert i == 0
                    system = message["content"]
                    continue
                if message["role"] == split_role and round:
                    rounds.append(round)
                    round = []
                round.append(message)
            if round:
                rounds.append(round)
            return system, rounds

        max_new_tokens = 500 or model.generation_config.max_new_tokens
        assert max_new_tokens is not None
        max_length = 2000 or model.config.max_length
        assert max_length is not None
        max_input_tokens = max_length - max_new_tokens
        system, rounds = _parse_messages(messages, split_role="user")
        system_tokens = tokenizer.encode(system)
        max_history_tokens = max_input_tokens - len(system_tokens)

        history_tokens = []
        for round in rounds[::-1]:
            round_tokens = []
            for message in round:
                if message["role"] == "user":
                    round_tokens.append(tokenizer.convert_tokens_to_ids("user")) # model.generation_config
                else:
                    round_tokens.append(tokenizer.convert_tokens_to_ids("assistant"))
                round_tokens.extend(tokenizer.encode(message["content"]))
            if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
                history_tokens = round_tokens + history_tokens  # concat left
                if len(history_tokens) < max_history_tokens:
                    continue
            break

        input_tokens = system_tokens + history_tokens
        if messages[-1]["role"] != "assistant":
            input_tokens.append(tokenizer.convert_tokens_to_ids("assistant")) #model.generation_config.assistant_token_id
        input_tokens = input_tokens[-max_input_tokens:]  # truncate left
        return torch.LongTensor([input_tokens])
            

    def cal_attn(self, part_id: int, question: str, reason: str, answer: str, msg: list) -> str:
        question_len = len(self.tokenizer(question, return_tensors="pt").input_ids[0])
        question_msg = msg
        question_ids = self.build_chat_input(self.model, self.tokenizer, question_msg)
        prompt_len = len(question_ids[0]) - question_len - 1
        question_len = len(question_ids[0])
        assistant_msg = [{'role':'assistant', 'content':f'Reason:{reason}\nAnswer: {answer}'}]
        input_msg = question_msg + assistant_msg
        input_ids = self.build_chat_input(self.model, self.tokenizer, input_msg)  
        stem_len = len(self.tokenizer(question.split('\n')[0],return_tensors="pt").input_ids[0])
        stem_len = prompt_len + stem_len
        cot_msg = question_msg + [{'role':'assistant', 'content':f'{reason}'}]
        cot_len = len(self.build_chat_input(self.model, self.tokenizer, input_msg)[0]) # Get Reason - Answer paired output
        
        self.model.eval()

        outputs = self.model(
            input_ids=input_ids.to(self.model.device),
=======
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
                token = self.tokenizer.batch_decode(chat_ids)[task_idx].strip()
                for token_ in nlp(token):
                    if token_.lemma_ not in self.scenery_words:
                        stop_words_ids.append(task_idx)
        return stop_words_ids

    def get_sentence_span_idx(
        self, chat_ids: torch.LongTensor
    ) -> list[tuple[int, int]]:
        """
        Get tuple of start, stop indices of period from the current task chat ids.

        :param chat_ids: current sample part ids (task and model's output)
        :return: list of indices of periods in the current task
        """
        period_token_id = self.tokenizer.encode(".", add_special_tokens=False)[0]
        period_token_indices = [
            i for i, tok in enumerate(chat_ids) if tok == period_token_id
        ]
        (
            period_token_indices.pop(-1)
            if period_token_indices[-1] == chat_ids[-1]
            else None
        )
        # Create start, stop tuples of indices
        period_token_indices = [
            (lambda x: (start, stop))((start, stop))
            for start, stop in zip([0] + period_token_indices, period_token_indices)
        ]

        return period_token_indices[1:]

    @staticmethod
    def get_attention_scores(
        output_tensor: torch.LongTensor,
        model_output_len: int,
        period_indices: list = None,
    ) -> np.ndarray:
        """
        Obtains the attention scores from a tensor of attention weights of the current chat.
        The function calculates the attention scores for current task tokens by averaging over layers,
        heads and normalizing over the sum of all token attention scores.

        (The following code is an adjusted version of the original implementation from Li et. al 2024
         (Link to paper: https://arxiv.org/abs/2402.18344))

        :param output_tensor: model output tensor
        :param model_output_len: model output length
        :param period_indices: indices of periods in the current task

        :return: 2D normalized attention scores averaged over layers and heads for the tokens of the current task
        #TODO Add check for empty model output
        #TODO Add supporting token markers
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

        if period_indices:
            # Additionally take mean of attention scores over each task sentence.
            # attn_scores = np.array([attn_scores.T[start:stop] for start, stop in period_indices]).squeeze().mean(axis=-1)
            attn_scores = np.array(
                [
                    attn_scores[:, start:stop].mean(axis=-1)
                    for start, stop in period_indices
                ]
            ).squeeze()
            # Reshape to match expected output format
            attn_scores_T = attn_scores.transpose(1, 0)
            # Normalize the attention scores by the sum of all token attention scores
            attn_scores_T = attn_scores_T / attn_scores_T.sum(axis=0, keepdims=True)

            assert attn_scores_T.shape == (attn_scores_T.shape[0], len(period_indices))
            return attn_scores_T

        return attn_scores

    def filter_attention_scores(
        self, attention_scores: np.ndarray, chat_ids: np.ndarray
    ) -> tuple[np.ndarray, list]:
        """
        Filter context and question attention scores for scenery words
        by their indices in each row of the output attention scores.
        Removes message role tokens.

        :param attention_scores: The attention scores of the current chat
        :param chat_ids: current sample part ids (task and model's output)
        :return: filtered attention scores with the according attention_indices
        """
        stop_words_indices = self.get_stop_word_idxs(attention_scores, chat_ids)
        attention_indices = list(
            filter(
                lambda x: x not in stop_words_indices, range(attention_scores.shape[1])
            )
        )
        return attention_scores[:, attention_indices], attention_indices

    def get_attention(
        self, part: SamplePart, chat: Chat, after: bool = True
    ) -> InterpretabilityResult:
        """
        1. Defines structural parts of the current chat and gets their input ids and lengths.
        2. Gets the relevant attention scores, filters them.
        3. Constructs x and y tokens and optionally creates heatmaps.

        (The following code is an adjusted version of the original implementation from Li et. al 2024
         (Link to paper: https://arxiv.org/abs/2402.18344))

        :param part: part of the sample with the output before the setting is applied
        :param chat: Chat history as list of messages
        :param after: if to get attention scores after the setting was applied to the model output or before
        :return: attention scores, tokenized x and y tokens
        """
        if (after and not part.result_after.model_answer) or not (
            after or part.result_before.model_answer
        ):
            raise ValueError("Interpretability called on empty model output")

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
>>>>>>> e6037e4ebd8ce8f2ac07f8a9c529eac953b0d5ef
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )

<<<<<<< HEAD
        print("CoT length: ", cot_len-prompt_len) #121
        print("Question length: ", question_len-prompt_len) #84
        
        input_ids = input_ids[0, prompt_len:].detach().cpu().numpy()
        scores = []

        attn_values = torch.stack(outputs['attentions'], dim=0).squeeze(1) # Tensor of attention weights
        attn_values = attn_values.mean(dim=0) # Take mean across layers
        attn_scores = attn_values[:, prompt_len:, prompt_len:].detach().cpu().numpy() #w/o system prompt
        
        attn_scores = attn_scores[:, question_len-prompt_len:cot_len-prompt_len, :question_len-prompt_len].mean(axis=0) # Mean over 40 heads
        attn_scores = attn_scores/attn_scores.sum(axis=-1, keepdims=True) # Normalize attention over tokens

        # Sum attention scores for each token over all Cot tokens to get top k cand. 
        top_k_attn_indices = np.argpartition(attn_scores.sum(axis=0), -20, axis=0)[-20:].squeeze() # Take top attended word from options given CoT (1, 10)
        np.sort(top_k_attn_indices) # Sort in ascending order
        
        top_k_attn_scores = attn_scores[:, top_k_attn_indices] # Take top attended word from options given CoT (41, 10)
        np.sort(top_k_attn_scores) # Sort ascending

        dist_categories = ["subj", "obj", "relation", "location", "nh-subj", "subj_attr", "attributes", "other"]
        dist_toks = []
        for cat in dist_categories:
            with open (f"{os.environ.get('INTERP_DIR')+cat}.txt", "r") as f:
                for line in f.readlines():
                    dist_toks.append(line.strip())
    
        print("Attn scores shape: ", attn_scores.shape)
        not_att = []
        for i, cot_tok in enumerate(attn_scores):
            for ind, attn in enumerate(attn_scores[i]): # Attention for first Cot token - top_k_…; attn, ind; 
                tok = self.tokenizer.batch_decode(input_ids)[ind] # convert ids to natural text.
                tok = tok.strip()
                if re.match("([^A-Za-z]+)|INST", tok) or not self.check_tok(dist_toks, tok): # Remove unneccessary tokens
                    not_att.append(ind)
        
        top_k_attn_indices = list(filter(lambda x: x not in not_att, list(range(attn_scores.shape[1])))) #top_k_attn_indices

        top_k_attn_scores = attn_scores[:, top_k_attn_indices]

        y_tokens = self.tokenizer.batch_decode(input_ids[question_len-prompt_len:cot_len-prompt_len])
        x_tokens = self.tokenizer.batch_decode(input_ids[top_k_attn_indices]) # Take options

        del attn_values, attn_scores, input_ids
        torch.cuda.empty_cache()
        del outputs
        torch.cuda.empty_cache()

        #return top_k_attn_scores, x_tokens, y_tokens
        fig_path = os.path.join("./results", f'task-{self.task_id}-{part_id}.pdf')

        plotter = Plotter(fig_path)
        plotter._draw_heat(index=x_tokens, scores=top_k_attn_scores, x=x_tokens, y=y_tokens, path=fig_path)

    
=======
        chat_ids = chat_ids[0, :].detach().cpu().numpy()

        # Check task length
        period_indices = self.get_sentence_span_idx(chat_ids)
        overflow = True if len(period_indices) >= 10 else False

        # Obtain attention scores from model output
        attention_scores = self.get_attention_scores(
            output_tensor=output_tensor,
            model_output_len=model_output_len,
            period_indices=period_indices if overflow else None,
        )

        if not overflow:
            attention_scores, attention_indices = self.filter_attention_scores(
                attention_scores, chat_ids
            )
            # Decode the task tokens
            x_tokens = self.tokenizer.batch_decode(chat_ids[attention_indices])
        else:
            x_tokens = []
            for inx in range(1, len(period_indices) + 1):
                if inx in part.supporting_sent_inx:
                    x_tokens.append(f"* {inx} *")
                else:
                    x_tokens.append(f"{inx}")

        # Decode the model output tokens without "assistant" token
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

        return InterpretabilityResult(attention_scores, x_tokens, y_tokens)
>>>>>>> e6037e4ebd8ce8f2ac07f8a9c529eac953b0d5ef
