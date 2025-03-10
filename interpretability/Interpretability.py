
from __future__ import annotations

import os
import re
from typing import List
import numpy as np
import torch

from plots.Plotter import Plotter
from settings.Model import Model
from interpretability.utils import Helper


class Interpretability:
    def __init__(self, model: Model = None, plotter: Plotter = None):
        """
        Interpretability class
        :param model: instance of Model
        :param interpr_path: interpretability path
        :param plotter: instance of Plotter
        """

        self.model = model.model
        self.tokenized = model.tokenizer
        self.plotter = plotter

        self.helper = Helper() 
        
    def parse_messages(self, messages, split_role="user"):
        """
        Parses messages by appending new message in message_round list to message_rounds after initial system message.
        :param messages: prompt messages 
        :param split_role: one of system/user
        :return: system message and message_rounds list
        """
        system_msg, message_rounds = "", []
        message_round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system_msg = message["content"]
                continue
            if message["role"] == split_role and message_round:
                message_rounds.append(message_round)
                message_round = []
            message_round.append(message)
        if message_round:
            message_rounds.append(message_round)
        return system_msg, message_rounds


    def build_chat_input(self, messages: List[dict], max_new_tokens: int=0):
        """
        As taken by CoT repo; builds chat input by concatenating it left-wise to current chat history.
        :param messages: List containing chat hisory
        :param max_new_tokens: max_new_tokens model config
        :param max_length: default max_length of model config 
        :return: tensor of input tokens
        """
        max_new_tokens = 500 or self.model.max_new_tokens
        max_length = 200 or self.model.config.max_length
        max_input_tokens = max_length - max_new_tokens
        system, rounds = self.parse_messages(messages, split_role="user")
        system_tokens = self.tokenized.encode(system)
        max_history_tokens = max_input_tokens - len(system_tokens)

        history_tokens = []
        for round in rounds[::-1]:
            round_tokens = []
            for message in round:
                if message["role"] == "user":
                    round_tokens.append(self.tokenized.convert_tokens_to_ids("user"))
                else:
                    round_tokens.append(self.tokenized.convert_tokens_to_ids("assistant"))
                round_tokens.extend(self.tokenized.encode(message["content"]))
            if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
                history_tokens = round_tokens + history_tokens  # concat left
                if len(history_tokens) < max_history_tokens:
                    continue
            break

        input_tokens = system_tokens + history_tokens
        if messages[-1]["role"] != "assistant":
            input_tokens.append(self.tokenized.convert_tokens_to_ids("assistant")) #model.generation_config.assistant_token_id
        input_tokens = input_tokens[-max_input_tokens:]  # truncate left
        return torch.LongTensor([input_tokens])
            

    def parse_chat(self, question: str, reasoning: str, answer: str, chat: list):
        
        """
        Parses chat by providing contents and lengths of its parts.
        :param question: model question
        :param reasoning: model reasoninging
        :param answer: model answer
        :param chat: chat history
        :return: info dict of chat parts
        """
        question_len = len(self.tokenized(question, return_tensors="pt").input_ids[0])
        question_msg = chat
        question_ids = self.build_chat_input(question_msg)
        prompt_len = len(question_ids[0]) - question_len - 1
        question_len = len(question_ids[0])
        assistant_msg = [{'role':'assistant', 'content':f'Reasoning:{reasoning}\nAnswer: {answer}'}]
        input_msg = question_msg + assistant_msg
        question_len = question_len - prompt_len

        stem_len = len(self.tokenized(question.split('\n')[0],return_tensors="pt").input_ids[0])
        stem_len = prompt_len + stem_len
        cot_msg = question_msg + [{'role':'assistant', 'content':f'{reasoning}'}]
        cot_len = len(self.build_chat_input(input_msg)[0]) # Get reasoning - Answer paired output
        cot_len = cot_len - prompt_len
        
        return {"question_len": question_len, "prompt_len": prompt_len, "input_msg": input_msg, "cot_len": cot_len}

    def get_attention_scores(self, outputs, prompt_len, question_len, cot_len) -> np.ndarray:
        """
        Obtains the attention scores from model output attention weights tensor for question and CoT tokens by averaging over layers, heads and normalizing over the sum of all token attention scores.

        :param outputs: model output
        :param prompt_len: prompt length
        :param question_len: question length
        :param cot_len: CoT length
        :param attn_scores: 2D normalized attention scores averaged over layers and heads for CoT and question tokens.
        """
        attn_tensor = torch.stack(outputs['attentions'], dim=0).squeeze(1)
        attn_tensor = attn_tensor.mean(dim=0)
        attn_scores = attn_tensor[:, prompt_len:, prompt_len:].float().detach().cpu().numpy() # w/o system prompt
        
        attn_scores = attn_scores[:, question_len:cot_len, :question_len].mean(axis=0)
        attn_scores = attn_scores/attn_scores.sum(axis=-1, keepdims=True)

        return attn_scores

    
    def calculate_attention(self, task_id: int, part_id: int, question: str, reasoning: str, answer: str, chat: list) -> dict:
        """
        Taken by CoT repo with changes:  Obtains attention scores through output attention weights as scores for CoT and Question for each part answer.
        :param task_id: Current task id
        :param part_id: Current sample part id
        :param question: The question of sample part
        :param reasoning: According reasoning of sample part
        :param answer: According model answer                                                                                                        
        :chat: Chat history as list of messages
        :return: attention scores, tokenized x and y tokens
        """
        input_parts = self.parse_chat(question, reasoning, answer, chat)
        
        input_msg = input_parts["input_msg"]
        question_len = input_parts["question_len"]
        prompt_len = input_parts["prompt_len"]
        cot_len = input_parts["cot_len"]

        input_ids = self.build_chat_input(input_msg)  

        outputs = self.model(
            input_ids=input_ids.to(self.model.device),
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )

        input_ids = input_ids[0, prompt_len:].detach().cpu().numpy()
        
        attn_scores = self.get_attention_scores(outputs, prompt_len, question_len, cot_len)

        not_att = []
        for i, cot_row in enumerate(attn_scores):
            for ind, attn in enumerate(cot_row):
                tok = self.tokenized.batch_decode(input_ids)[ind] 
                tok = tok.strip()
                if self.helper.is_stop_word(tok):
                    not_att.append(ind)
        
        attn_indices = list(filter(lambda x: x not in not_att, range(attn_scores.shape[1])))

        attn_scores = attn_scores[:, attn_indices]

        y_tokens = self.tokenized.batch_decode(input_ids[question_len:cot_len]) 
        x_tokens = self.tokenized.batch_decode(input_ids[attn_indices]) 

        self.plotter.draw_heat(index=x_tokens, x=x_tokens, y=y_tokens, scores=attn_scores, task_id=task_id, part_id=part_id)
        torch.cuda.empty_cache()

        interpr_results = {"attn_scores": attn_scores, "x_tokens": x_tokens, "y_tokens": y_tokens}

        return interpr_results
   