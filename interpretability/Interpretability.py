from __future__ import annotations

import torch

from interpretability.utils import get_attn_toks, is_stop_word
from plots.Plotter import Plotter
from prompts.Chat import Chat, SamplePart
from settings.Model import Model


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

    def get_attention_scores(self, outputs, prompt_len, question_len, cot_len):
        """
        Obtains the attention scores from model output attention weights tensor for question and CoT tokens by averaging over layers, heads and normalizing over the sum of all token attention scores.

        :param outputs: model output
        :param prompt_len: prompt length
        :param question_len: question length
        :param cot_len: CoT length
        :param attn_scores: 2D normalized attention scores averaged over layers and heads for CoT and question tokens.
        """
        attn_tensor = torch.stack(outputs["attentions"], dim=0).squeeze(1)
        print(attn_tensor)
        attn_tensor = attn_tensor.mean(dim=0)
        attn_scores = (
            attn_tensor[:, prompt_len:, prompt_len:].float().detach().cpu().numpy()
        )  # w/o system prompt
        print("SHAPE: ", attn_scores.shape, question_len, cot_len, prompt_len)
        attn_scores = attn_scores[:, question_len:cot_len, :question_len].mean(axis=0)
        print("SHAPE2: ", attn_scores.shape)
        attn_scores = attn_scores / attn_scores.sum(axis=-1, keepdims=True)
        print("SHAPE3", attn_scores.shape)
        assert attn_scores is not None
        print("SXOC", attn_scores)
        return attn_scores

    def calculate_attention(self, part: SamplePart, chat: Chat):
        """
        Taken by CoT repo with changes:  Obtains attention scores through output attention weights as scores for CoT and Question for each part answer.
        :param part: part instance of the sample
        :param chat: Chat history as list of messages
        :return: attention scores, tokenized x and y tokens
        """

        input_parts = chat.interpr_parse_chat(
            part.question,
            part.reasoning,
            part.answer,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        input_msg = input_parts["input_msg"]
        question_len = input_parts["question_len"]
        prompt_len = input_parts["prompt_len"]
        cot_len = input_parts["cot_len"]

        input_ids = chat.interp_build_chat_input(
            messages=input_msg, tokenizer=self.tokenizer
        )
        print("ids: ", input_ids)
        outputs = self.model(
            input_ids=input_ids.to(self.model.device),
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )
        # print("out:", outputs)
        input_ids = input_ids[0, prompt_len:].detach().cpu().numpy()
        print("ids: ", input_ids)
        attn_scores = self.get_attention_scores(
            outputs, prompt_len, question_len, cot_len
        )
        print("scores", attn_scores)
        # Obtain attention tokens
        attn_tokens = get_attn_toks()

        not_att = []
        for cot_row in attn_scores:
            print("cot_row", cot_row)
            for ind, attn in enumerate(cot_row):
                tok = self.tokenizer.batch_decode(input_ids)[ind]
                tok = tok.strip()
                print(tok)
                if is_stop_word(tok, attn_tokens):
                    not_att.append(ind)

        attn_indices = list(
            filter(lambda x: x not in not_att, range(attn_scores.shape[1]))
        )

        attn_scores = attn_scores[:, attn_indices]

        print(attn_indices, attn_scores)
        y_tokens = self.tokenizer.batch_decode(input_ids[question_len:cot_len])
        x_tokens = self.tokenizer.batch_decode(input_ids[attn_indices])

        self.plotter.draw_heat(
            index=x_tokens,
            x=x_tokens,
            y=y_tokens,
            scores=attn_scores,
            task_id=part.task_id,
            sample_id=part.sample_id,
            part_id=part.part_id,
        )
        torch.cuda.empty_cache()

        # TO DISCUSS: would be better to save the interpretability in files
        # interpr_results = {
        #     "attn_scores": attn_scores,
        #     "x_tokens": x_tokens,
        #     "y_tokens": y_tokens,
        # }
        #
        # return interpr_results
