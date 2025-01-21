from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(Path.cwd()).parents[0]))


@dataclass
class Source:
    """
    This class handles the roles of the participants in the conversation.
    """

    system = "system"
    user = "user"
    assistant = "assistant"


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
        return {"role": role, "content": part}

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
