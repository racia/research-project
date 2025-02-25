from __future__ import annotations

import settings.utils as utils
from prompts.Chat import Chat
from prompts.Prompt import Prompt
from settings.Model import Model
from settings.Setting import Setting
from settings.utils import Enumerate


class Baseline(Setting):
    """
    The baseline class.
    """

    def __init__(
        self,
        model: Model,
        to_enumerate: dict[Enumerate, bool],
        total_tasks: int,
        total_parts: int,
        samples_per_task: int = 5,
        prompt: Prompt = None,
    ):
        """
        Baseline class manages model runs and data flows around it.

        :param model: the model to use
        :param to_enumerate: dictionary with the settings to enumerate
        :param total_tasks: total number of tasks
        :param total_parts: total number of parts
        :param samples_per_task: number of samples per task for logging
        :param prompt: system prompt to start conversations
        """
        self.model = model

        self.prompt = prompt
        self.to_enumerate = to_enumerate

        self.question_id = 0
        self.total_tasks = total_tasks
        self.total_parts = total_parts
        self.samples_per_task = samples_per_task

    def prepare_prompt(self, chat: Chat) -> str:
        """
        Prepares the prompt to include the current part of the sample.

        :param chat: the chat object
        :return: the formatted prompt with the chat template applied
        """
        if self.model.to_continue:
            formatted_prompt = self.model.tokenizer.apply_chat_template(
                chat.messages, tokenize=False, continue_final_message=True
            )
        else:
            formatted_prompt = self.model.tokenizer.apply_chat_template(
                chat.messages, tokenize=False, add_generation_prompt=True
            )
        print(
            "Formatted prompt:",
            formatted_prompt,
            sep="\n",
            end="\n",
        )

        return formatted_prompt

    def apply_setting(self, decoded_output: str) -> dict[str, str]:
        """
        Postprocesses the output of the model.
        For the baseline model, this postprocessing just parses the output.

        :param decoded_output: the decoded output
        :return: dictionary with either the parsed model answer or just the model answer
        """
        parsed_output = utils.parse_output(output=decoded_output)
        return parsed_output
