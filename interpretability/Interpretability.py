from __future__ import annotations

import torch

from inference.Chat import Chat, SamplePart
from plots.Plotter import Plotter
from prompts.Prompt import Prompt
from settings.Model import Model
from interpretability.utils import get_stop_words
from transformers import AutoTokenizer


class Interpretability:
    def __init__(self, model: Model = None, plotter: Plotter = None):
        """
        Interpretability class
        :param model: instance of Model
        :param interpr_path: interpretability path
        :param plotter: instance of Plotter
        """
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.plotter = plotter
        self.max_new_tokens = model.max_new_tokens
    
            
    def get_attention_scores(self, outputs, prompt_len, question_len, cot_len) -> np.ndarray:
        """
        (Taken by CoT repo) Obtains the attention scores from model output attention weights tensor for question and CoT tokens by averaging over layers, heads and normalizing over the sum of all token attention scores.

        :param outputs: model output
        :param prompt_len: prompt length
        :param question_len: question length
        :param cot_len: CoT length
        :param attn_scores: 2D normalized attention scores averaged over layers and heads for cot and question tokens.
        """
        attn_tensor = torch.stack(outputs['attentions'], dim=0).squeeze(1)
        attn_tensor = attn_tensor.mean(dim=0) # Mean over layers
        attn_scores = attn_tensor[:, prompt_len:, prompt_len:].float().detach().cpu().numpy() # w/o system prompt
        attn_scores = attn_scores[:, question_len:cot_len, :question_len].mean(axis=0)  # Mean over heads
        attn_scores = attn_scores/attn_scores.sum(axis=-1, keepdims=True) # Normalize over all tokens

        return attn_scores
    


    def calculate_attention(self, part: SamplePart, chat: Chat) ->  dict[list]:
        """
        (Taken by CoT repo) Obtains attention scores through output attention weights as scores for CoT and question for each part answer.
        :param task_id: Current task id
        :param part_id: Current sample part id
        :param question: The question of sample part
        :param reasoning: According reasoning of sample part
        :param answer: According model answer                                                                                                        
        :chat: Chat history as list of messages
        :return: attention scores, tokenized x and y tokens
        """

        task_id = part.task_id
        sample_id = part.sample_id
        part_id = part.part_id
        
        # Obtain input_ids and lengths of current part
        curr_task_ids, curr_task_len = chat.get_ids_and_lengths(chat.messages[-2], max_new_tokens=self.max_new_tokens, tokenizer=self.tokenizer)
        prev_hist_ids, prev_hist_len = chat.get_ids_and_lengths(chat.messages[:-3], max_new_tokens=self.max_new_tokens, tokenizer=self.tokenizer)
        input_ids, input_len = chat.get_ids_and_lengths(chat.messages[:-2], max_new_tokens=self.max_new_tokens, tokenizer=self.tokenizer)


        # Feed to the model
        outputs = self.model(
            input_ids=input_ids.to(self.model.device),
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )

        # Obtain attention scores from model output 
        input_ids = input_ids[0, prev_hist_len:].detach().cpu().numpy()
        attn_scores = self.get_attention_scores(outputs, prev_hist_len, curr_task_len, input_len)

        # Filter for most import model answer tokens
        stop_words_indices = get_stop_words(self.tokenizer, input_ids, attn_scores)
        attn_indices = list(filter(lambda x: x not in stop_words_indices, range(attn_scores.shape[1])))
        attn_scores = attn_scores[:, attn_indices] # Shape: (Cot; Y_tokens i.e. indices)
        
        # Decode
        y_tokens = self.tokenizer.batch_decode(input_ids[curr_task_len:input_len]) #Cot tokens
        x_tokens = self.tokenizer.batch_decode(input_ids[attn_indices]) # model answer tokens

        # Call plotter
        self.plotter.draw_heat(index=x_tokens, x=x_tokens, y=y_tokens, scores=attn_scores, task_id=task_id, sample_id=sample_id, part_id=part_id)
        torch.cuda.empty_cache()

        # TO DISCUSS: would be better to save the interpretability in files
        interpr_results = {
            "attn_scores": attn_scores,
            "x_tokens": x_tokens,
            "y_tokens": y_tokens,
        }
        
        return interpr_results
