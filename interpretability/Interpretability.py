from __future__ import annotations

import torch

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
    
    
    def get_input_part_info(self, part_result:dict, chat:Chat) -> dict[int]:
        """
        Creates input_ids from the current part and provides length info for different parts.
        :param part_result: The dict containing current part
        :param tokenizer: The Model tokenizer
        :param chat: Current chat instance
        :return: A dictionary containing the necessary info on parts for calculating attention
        """
        question = part_result["part"]
        reasoning = part_result["model_reasoning"]
        answer = part_result["model_answer"]

        question_msg = [chat.messages[-2]]
        question_len = len(self.tokenizer(question, return_tensors="pt").input_ids[0])
        question_ids = chat.interp_build_chat_input(question_msg, tokenizer=self.tokenizer)

        # @TODO: Use our code
        # prompt = Prompt()
        # formatted_question = prompt.format_part(part=question, to_enumerate=True)
        # formatted_question_len = len(self.tokenizer(formatted_question, return_tensorts="pt").input_ids[0])

        prompt_len = len(question_ids[0]) - question_len - 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        question_len = len(question_ids[0])

        assistant_msg = chat.format_message(part=f"Reasoning: {reasoning}\nAnswer: {answer}", role="assistant")
        input_msg = question_msg + [assistant_msg]
        input_ids = chat.interp_build_chat_input(input_msg, tokenizer=self.tokenizer)
        cot_len = len(input_ids[0])
        
        return {"input_ids": input_ids, "question_len": question_len, "cot_len": cot_len, "prompt_len": prompt_len}
    

    def calculate_attention(self, part_result: dict, chat: Chat) ->  dict[List]:
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

        task_id = part_result["task_id"]
        sample_id = part_result["sample_no"]
        part_id = part_result["part_id"]
        
        # Obtain input_ids and lengths of current part
        ids_dict = self.get_input_part_info(part_result=part_result, chat=chat)
        input_ids = ids_dict["input_ids"]

        # Obtain info on part lengths
        question_len = ids_dict["question_len"]
        cot_len = ids_dict["cot_len"]
        prompt_len = ids_dict["prompt_len"]
        
        # Substract prompt_len to obtain actual length of current part
        question_len = question_len - prompt_len
        cot_len = cot_len - prompt_len

        # Feed to the model
        outputs = self.model(
            input_ids=input_ids.to(self.model.device),
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )

        # Obtain attention scores from model output 
        input_ids = input_ids[0, prompt_len:].detach().cpu().numpy()
        attn_scores = self.get_attention_scores(outputs, prompt_len, question_len, cot_len)

        # Filter for most import model answer tokens
        stop_words_indices = get_stop_words(self.tokenizer, input_ids, attn_scores)
        attn_indices = list(filter(lambda x: x not in stop_words_indices, range(attn_scores.shape[1])))
        attn_scores = attn_scores[:, attn_indices] # Shape: (Cot; Y_tokens i.e. indices)
        
        # Decode
        y_tokens = self.tokenizer.batch_decode(input_ids[question_len:cot_len]) #Cot tokens
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
