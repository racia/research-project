from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
from transformers import AutoTokenizer

from inference.DataLevels import SamplePart
#from inference.Prompt import Prompt
from inference.utils import generation_token


@dataclass
class Source:
    """
    This class handles the roles of the participants in the conversation.
    """

    system: str = "system"
    user: str = "user"
    assistant: str = "assistant"

    options = (system, user, assistant)


class Chat:
    """
    This class handles the chats with the model.
    """

    def __init__(self, system_prompt: Prompt = None, multi_system: bool = False):
        """
        Create a chat.
        A chat consists of the prompts the model is prompted with and the answers of the model.
        The prompts and answers are saved in a list.

        :param system_prompt: the first prompt the model is prompted with
        :param multi_system: whether the chat for one sample consists of multiple systems, i.e. a teacher and a student
        """
        self.multi_system: bool = multi_system

        if multi_system:
            self.messages = {
                "teacher": [],
                "student": [
                    {
                        "role": Source.system,
                        "content": system_prompt.text,
                        "original_content": system_prompt.original_text,
                    }
                ],
            }
        else:
            self.messages = [
                {
                    "role": Source.system,
                    "content": system_prompt.text,
                    "original_content": system_prompt.original_text,
                }
            ]

    @staticmethod
    def format_message(
        part: SamplePart | str, role: Union[Source.user, Source.assistant]
    ) -> dict[str, str]:
        """
        Formats the prompt by managing the data type and putting in into
        a dictionary the model expects.

        :param part: part of a sample as a string or a list of strings
        :param role: the producer of the message
        :return: prompt formatted as a dict
        """

        if isinstance(part, str):
            return {
                "role": role,
                "content": part,
                "original_content": part,
            }
        return {
            "role": role,
            "content": part.task,
            "original_content": part.unwrapped_task,
        }

    def add_message(
        self,
        part: SamplePart | str,
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

    def convert_into_ids(
        self,
        tokenizer: AutoTokenizer,
        chat_part: list[dict] = None,
        max_new_tokens: int = 100,
        max_length: int = 2048,
    ) -> tuple[torch.LongTensor, list[tuple[int, int]]]:
        """
        Converts either all the chat messages or the specified ones into ids ensuring that the input does not exceed
        the max_length. The system prompt is always included in the input, regardless of the chat_part.
        The assistant token id is always added at the end of the input.

        (Partly taken from https://arxiv.org/abs/2402.18344)

        :param tokenizer: tokenizer to use
        :param chat_part: chat part to convert into ids, if None, all messages are used
        :param max_new_tokens: max_new_tokens model config
        :param max_length: default max_length of model config
        :return: tensor of input tokens
        """
        input_tokens_left = max_length - max_new_tokens

        history_ids = []
        supporting_sent_spans = []
        for message in chat_part if chat_part else self.messages:
            message_ids = [generation_token(tokenizer, message["role"])]

            for sentence in message["original_content"].split("\n"):
                tokenized_sentence = tokenizer.encode(
                    sentence, add_special_tokens=False
                )
                if message["role"] == "user":
                    start = len(message_ids) + 1
                    message_ids.extend(tokenized_sentence)
                    end = len(message_ids)
                    supporting_sent_spans.append((start, end))
                else:
                    message_ids.extend(tokenized_sentence)

            if len(history_ids) + len(message_ids) <= input_tokens_left:
                history_ids += message_ids
            elif message["role"] == "assistant":
                history_ids.append(tokenizer.convert_tokens_to_ids("assistant"))
                break
            elif message["role"] == "user":
                break
            else:
                raise Exception("Unexpected error for message:", message)

        # take all the tokens that could fit
        return (
            torch.LongTensor([history_ids[-input_tokens_left:]]),
            supporting_sent_spans,
        )
