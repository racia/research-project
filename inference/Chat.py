from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
from transformers import PreTrainedTokenizerFast

from inference.DataLevels import SamplePart
from inference.Prompt import Prompt
from inference.utils import generation_tokens


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
        tokenizer: PreTrainedTokenizerFast,
        chat_part: list[dict] = None,
        max_length: int = 2048,
    ) -> tuple[dict[str, torch.LongTensor], list[tuple[int, int]], int]:
        """
        Converts either all the chat messages or the specified ones into ids ensuring that the input does not exceed
        the max_length. The system prompt is always included in the input, regardless of the chat_part.
        The assistant token id is always added at the end of the input.

        (Partly taken from https://arxiv.org/abs/2402.18344)

        :param tokenizer: tokenizer to use
        :param chat_part: chat part to convert into ids, if None, all messages are used
        :param max_length: default max_length of model config
        :return: tensor of input tokens, supporting sentence spans and system prompt length
        """
        sys_prompt_len = 0
        history_ids = []
        supporting_sent_spans = []
        for i, message in enumerate(chat_part if chat_part else self.messages):
            message_ids = generation_tokens(
                tokenizer, message["role"], eot=False if i == 0 else True
            )

            for sentence in message["original_content"].split("\n"):
                # \n\n in source produces empty sentences
                if sentence:
                    tokenized_sentence = tokenizer.encode(
                        sentence,
                        add_special_tokens=False,
                        return_tensors="pt",
                    )[0]
                    torch.cuda.empty_cache()
                    tokenized_sentence = tokenized_sentence.tolist()

                    if message["role"] == "user":
                        start = len(message_ids) + 1
                        message_ids.extend(tokenized_sentence)
                        end = len(message_ids)
                        supporting_sent_spans.append((start, end))
                    else:
                        message_ids.extend(tokenized_sentence)
            if message["role"] == "system":
                sys_prompt_len = len(message_ids)

            if len(history_ids) + len(message_ids) <= max_length:
                history_ids += message_ids
            else:
                break
            
        history_ids.append(tokenizer.convert_tokens_to_ids("assistant"))

        return (
            {"input_ids": torch.LongTensor([history_ids])},
            supporting_sent_spans,
            sys_prompt_len,
        )
