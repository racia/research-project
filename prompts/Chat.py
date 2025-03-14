from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import torch


@dataclass
class Source:
    """
    This class handles the roles of the participants in the conversation.
    """

    system = "system"
    user = "user"
    assistant = "assistant"


class SamplePart:
    def __init__(
        self,
        id_: int,
        task_id: int,
        sample_id: int,
        part_id: int,
        context: str,
        question: str,
        reasoning: str,
        answer: str,
        golden_answer: str,
        silver_reasoning=None,
        tokenizer=None,
    ):
        self.id_ = id_
        self.task_id = task_id
        self.sample_id = sample_id
        self.part_id = part_id

        self.context = context
        self.question = question
        self.reasoning = reasoning
        self.answer = answer

        self.golden_answer = golden_answer
        self.silver_reasoning = silver_reasoning

        self.model_output = None
        self.text = f"{context}\n{question}\n{reasoning}\n{answer}".strip()

        self.tokenizer = tokenizer

        if self.tokenizer:
            self.context_ids = self.tokenize(self.context)
            self.question_ids = self.tokenize(self.question)
            self.reasoning_ids = self.tokenize(self.reasoning)
            self.answer_ids = self.tokenize(self.answer)

            self.context_ids_len = len(self.context_ids)
            self.question_ids_len = len(self.question_ids)
            self.reasoning_ids_len = len(self.reasoning_ids)
            self.answer_ids_len = len(self.answer_ids)

    def tokenize(self, sentence: str):
        return self.tokenizer(sentence, return_tensors="pt").input_ids[0]

    def get_result(self):
        return {
            "id": self.id_,
            "task_id": self.task_id,
            "sample_id": self.sample_id,
            "part_id": self.part_id,
            "context": self.context,
            "question": self.question,
            "reasoning": self.reasoning,
            "answer": self.answer,
            "golden_answer": self.golden_answer,
            "silver_reasoning": self.silver_reasoning,
            "model_output": self.model_output,
        }


class Chat:
    """
    This class handles the chats with the model.
    """

    def __init__(self, system_prompt: str, multi_system: bool = False):
        """
        Create a chat.
        A chat consists of the prompts the model is prompted with and the answers of the model.
        The prompts and answers are saved in a list.

        :param system_prompt: the first prompt the model is prompted with
        :param multi_system: whether the chat for one sample consists of multiple systems, i.e. a teacher and a student
        """
        self.multi_system = multi_system

        if multi_system:
            self.messages = {
                "teacher": [],
                "student": [{"role": Source.system, "content": system_prompt}],
            }
        else:
            self.messages = [{"role": Source.system, "content": system_prompt}]

    @staticmethod
    def format_message(
        part: str | list[str], role: Union[Source.user, Source.assistant]
    ) -> dict[str, str]:
        """
        Formats the prompt by managing the data type and putting in into
        a dictionary the model expects.

        :param part: part of a sample as a string or a list of strings
        :param role: the producer of the message
        :return: prompt formatted as a dict
        """
        if type(part) is list:
            part = "\n".join(part)
        return {
            "role": role,
            "content": part,
        }

    def add_message(
        self,
        part: str | list[str],
        source: Union[Source.user, Source.assistant],
        model_role: str = "student",
    ) -> None:
        """
        Add a message to the messages list.

        :param part: part of a sample as a string or a list of strings
        :param source: the producer of the message
        :param model_role: the model that produced the message. Only necessary when using a multi-system chat
        """
        if self.multi_system:
            self.messages[model_role].append(self.format_message(part, source))
        else:
            self.messages.append(self.format_message(part, source))

    def interpr_parse_messages(self, messages, split_role="user"):
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

    def interp_build_chat_input(
        self,
        messages: List[dict],
        max_new_tokens: int = 100,
        max_length: int = 200,
        tokenizer=None,
    ):
        """
        As taken by CoT repo: Builds chat input by concatenating it left-wise to current chat messages excluding last concatenated model output.
        :param messages: List containing chat hisory
        :param max_new_tokens: max_new_tokens model config
        :param max_length: default max_length of model config
        :return: tensor of input tokens
        """
        max_input_tokens = max_length - max_new_tokens
        system, rounds = self.interpr_parse_messages(
            messages, split_role="user"
        )  # Exclude last model answer, i.e. assistant msg
        system_tokens = tokenizer.encode(system)
        max_history_tokens = max_input_tokens - len(system_tokens)

        history_tokens = []
        for round in rounds[::-1]:
            round_tokens = []
            for message in round:
                if message["role"] == "user":
                    round_tokens.append(tokenizer.convert_tokens_to_ids("user"))
                else:
                    round_tokens.append(tokenizer.convert_tokens_to_ids("assistant"))
                round_tokens.extend(tokenizer.encode(message["content"]))
            if (
                len(history_tokens) == 0
                or len(history_tokens) + len(round_tokens) <= max_history_tokens
            ):
                history_tokens = round_tokens + history_tokens  # concat left
                if len(history_tokens) < max_history_tokens:
                    continue
            break

        input_tokens = system_tokens + history_tokens
        if messages[-1]["role"] != "assistant":
            input_tokens.append(
                tokenizer.convert_tokens_to_ids("assistant")
            )  # model.generation_config.assistant_token_id
        input_tokens = input_tokens[-max_input_tokens:]  # truncate left
        return torch.LongTensor([input_tokens])

    def interpr_parse_chat(
        self, question: str, reasoning: str, answer: str, model, tokenizer
    ):
        """
        Parses chat by providing contents and lengths of its parts.
        :param question: model question
        :param reasoning: model reasoninging
        :param answer: model answer
        :param chat: chat history
        :return: info dict of chat parts
        """
        question_len = len(tokenizer(question, return_tensors="pt").input_ids[0])
        question_msg = self.messages[:-1]  # old chat is self.messages
        print("Q msg", question_msg)
        question_ids = self.interp_build_chat_input(question_msg, tokenizer=tokenizer)
        prompt_len = len(question_ids[0]) - question_len - 1
        question_len = len(question_ids[0])
        assistant_msg = [
            {
                "role": "assistant",
                "content": f"Reasoning: {reasoning}\nAnswer: {answer}",
            }
        ]
        input_msg = question_msg + assistant_msg
        question_len = question_len - prompt_len
        cot_msg = question_msg + [{"role": "assistant", "content": f"{reasoning}"}]
        cot_len = len(
            self.interp_build_chat_input(input_msg, tokenizer=tokenizer)[0]
        )  # Get reasoning - Answer paired output
        cot_len = cot_len - prompt_len
        print("COT", cot_len)
        return {
            "question_len": question_len,
            "prompt_len": prompt_len,
            "input_msg": input_msg,
            "cot_len": cot_len,
        }
