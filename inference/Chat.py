from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import torch
from transformers import AutoTokenizer


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

    def filter_messages(
        self, split_role="user", messages: list[dict] = None
    ) -> Tuple[str, list[str]]:
        """
        Filters messages by role and gets system message

        :param split_role: message role
        :param messages: list of messages to filter
        :return: system message and filtered messages
        """
        if not messages:
            messages = self.messages
        system_msg = self.messages[0]["content"]
        message_rounds = [m["content"] for m in messages[1:] if m["role"] == split_role]

        return system_msg, message_rounds

    def convert_into_ids(
        self,
        tokenizer: AutoTokenizer,
        chat_part: list[dict] = None,
        max_new_tokens: int = 100,
        max_length: int = 2048,
    ) -> torch.LongTensor:
        """
        Converts either all the chat messages or the specified ones into ids ensuring that the input does not exceed
        the max_length. The system prompt is always included in the input, regardless of the chat_part.
        The assistant token id is always added at the end of the input.

        :param tokenizer: tokenizer to use
        :param chat_part: chat part to convert into ids, if None, all messages are used
        :param max_new_tokens: max_new_tokens model config
        :param max_length: default max_length of model config
        :return: tensor of input tokens
        """
        input_tokens_left = max_length - max_new_tokens
        # TODO: Our case: add_special_tokens = True
        system_prompt_ids = tokenizer.encode(self.messages[0]["content"])
        max_history_len = input_tokens_left - len(system_prompt_ids)

        # take all the messages except the system prompt backwards if we go through all the messages

        history_ids = []
        for message in chat_part:
            message_ids = [generation_token(tokenizer, message["role"])]
            if message["role"] == "user":
                message_ids.append(tokenizer.convert_tokens_to_ids("user"))
            elif message["role"] == "assistant":
                message_ids.append(tokenizer.convert_tokens_to_ids("assistant"))
            else:
                raise ValueError(f"Unknown role: {message['role']}")
            
            message_ids.extend(tokenizer.encode(message["original_content" if message["role"]=="system" else "content"]))

            if len(history_ids) + len(message_ids) <= max_history_len:
                history_ids += message_ids
            elif message["role"] == "assistant":
                history_ids.append(tokenizer.convert_tokens_to_ids("assistant"))
                break
            elif message["role"] == "user":
                break
            else:
                raise Exception("Unexpected error for message:", message)

        input_tokens = system_prompt_ids + history_ids

        # take all the tokens that could fit
        input_tokens = system_prompt_ids + history_ids[-input_tokens_left:]
        return torch.LongTensor([input_tokens])
