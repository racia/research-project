
from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import List
import numpy as np
import torch

from plots.Plotter import Plotter
from settings.Model import Model


@dataclass
class Interpretability:

    model: Model
    path: str
    
    def __init__(self, interpretability: bool = False, model: Model = None, path: str = None):
        """
        Interpretability class
        :param interpretability: whether to instantiate the interpretability class
        :param model: Model object 
        :path: results saving path @TODO: Add to DataSaver
        """
        if interpretability:
            self.model = model
            self.path = path
        else: 
            return 

    def check_tok(self, tokens, tok):
        """
        Checks whether token is in set of relevant tokens .txt files
        :param dist_toks: token sets
        """
        return tok in tokens
      
    def parse_messages(self, messages, split_role="user"):
        """
        Parses messages by appending user content as new round to rounds after initial system content
        :param messages: prompt messages 
        :param split_role: system/user
        :return: system message and rounds list
        """
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


    def build_chat_input(self, messages: List[dict], max_new_tokens: int=0):
        """
        As taken by CoT repo.        
        """
        max_new_tokens = 500 or self.model.generation_config.max_new_tokens
        assert max_new_tokens is not None
        max_length = 2000 or self.model.config.max_length
        assert max_length is not None
        max_input_tokens = max_length - max_new_tokens
        system, rounds = self.parse_messages(messages, split_role="user")
        system_tokens = self.model.tokenizer.encode(system)
        max_history_tokens = max_input_tokens - len(system_tokens)

        history_tokens = []
        for round in rounds[::-1]:
            round_tokens = []
            for message in round:
                if message["role"] == "user":
                    round_tokens.append(self.model.tokenizer.convert_tokens_to_ids("user")) # model.generation_config
                else:
                    round_tokens.append(self.model.tokenizer.convert_tokens_to_ids("assistant"))
                round_tokens.extend(self.model.tokenizer.encode(message["content"]))
            if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
                history_tokens = round_tokens + history_tokens  # concat left
                if len(history_tokens) < max_history_tokens:
                    continue
            break

        input_tokens = system_tokens + history_tokens
        if messages[-1]["role"] != "assistant":
            input_tokens.append(self.model.tokenizer.convert_tokens_to_ids("assistant")) #model.generation_config.assistant_token_id
        input_tokens = input_tokens[-max_input_tokens:]  # truncate left
        return torch.LongTensor([input_tokens])
            

    def cal_attn(self, task_id: int, part_id: int, question: str, reason: str, answer: str, msg: list) -> str:
        """
        Taken by CoT repo with changes:

        Obtains attention values through output attention weights as scores for CoT and Question for each part answer.
        @TODO: Remove top_k_attetion since every token in answer is considered.
        """
        question_len = len(self.model.tokenizer(question, return_tensors="pt").input_ids[0])
        question_msg = msg
        question_ids = self.build_chat_input(question_msg)
        prompt_len = len(question_ids[0]) - question_len - 1
        question_len = len(question_ids[0])
        assistant_msg = [{'role':'assistant', 'content':f'Reason:{reason}\nAnswer: {answer}'}]
        input_msg = question_msg + assistant_msg
        input_ids = self.build_chat_input(input_msg)  
        stem_len = len(self.model.tokenizer(question.split('\n')[0],return_tensors="pt").input_ids[0])
        stem_len = prompt_len + stem_len
        cot_msg = question_msg + [{'role':'assistant', 'content':f'{reason}'}]
        cot_len = len(self.build_chat_input(input_msg)[0]) # Get Reason - Answer paired output
        
        self.model.model.eval()

        outputs = self.model.model(
            input_ids=input_ids.to(self.model.model.device),
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )

        print("CoT length: ", cot_len-prompt_len) #121
        print("Question length: ", question_len-prompt_len) #84
        
        input_ids = input_ids[0, prompt_len:].detach().cpu().numpy()
        scores = []

        attn_values = torch.stack(outputs['attentions'], dim=0).squeeze(1) # Tensor of attention weights
        attn_values = attn_values.mean(dim=0) # Take mean across layers
        attn_scores = attn_values[:, prompt_len:, prompt_len:].float().detach().cpu().numpy() #w/o system prompt
        
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
            with open (f"{self.path+cat}.txt", "r") as f:
                for line in f.readlines():
                    dist_toks.append(line.strip())
    
        print("Attn scores shape: ", attn_scores.shape)
        not_att = []
        for i, cot_tok in enumerate(attn_scores):
            for ind, attn in enumerate(attn_scores[i]): # Attention for first Cot token - top_k_…; attn, ind; 
                tok = self.model.tokenizer.batch_decode(input_ids)[ind] # convert ids to natural text.
                tok = tok.strip()
                if re.match("([^A-Za-z]+)|INST", tok) or not self.check_tok(dist_toks, tok): # Remove unneccessary tokens
                    not_att.append(ind)
        
        top_k_attn_indices = list(filter(lambda x: x not in not_att, list(range(attn_scores.shape[1])))) #top_k_attn_indices

        top_k_attn_scores = attn_scores[:, top_k_attn_indices]

        y_tokens = self.model.tokenizer.batch_decode(input_ids[question_len-prompt_len:cot_len-prompt_len])
        x_tokens = self.model.tokenizer.batch_decode(input_ids[top_k_attn_indices]) # Take options

        del attn_values, attn_scores, input_ids
        torch.cuda.empty_cache()
        del outputs
        torch.cuda.empty_cache()

        #return top_k_attn_scores, x_tokens, y_tokens
        fig_path = os.path.join(str(self.path), "results", f'task-{task_id}-{part_id}.pdf')

        plotter = Plotter(fig_path)
        plotter.draw_heat(index=x_tokens, scores=top_k_attn_scores, x=x_tokens, y=y_tokens, path=fig_path)

    