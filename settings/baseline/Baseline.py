from __future__ import annotations

import settings.utils as utils
from inference.Chat import Chat
from inference.Prompt import Prompt
from interpretability.Interpretability import Interpretability
from settings.Model import Model
from settings.Setting import Setting
from settings.config import Wrapper
from settings.utils import Enumerate


class Baseline(Setting):
    """
    The baseline class.
    """

    def __init__(
        self,
        model: Model,
        to_enumerate: Enumerate,
        total_tasks: int,
        total_parts: int,
        samples_per_task: int = 5,
        init_prompt: Prompt = None,
        wrapper: Wrapper = None,
        interpretability: Interpretability = None,
    ):
        """
        Baseline class manages model runs and data flows around it.

        :param model: the model to use
        :param to_enumerate: dictionary with the settings to enumerate
        :param total_tasks: total number of tasks
        :param total_parts: total number of parts
        :param samples_per_task: number of samples per task for logging
        :param init_prompt: system prompt to start conversations
        """
        super().__init__(
            model=model,
            total_tasks=total_tasks,
            total_parts=total_parts,
            samples_per_task=samples_per_task,
            init_prompt=init_prompt,
            to_enumerate=to_enumerate,
            wrapper=wrapper,
        )
        self.question_id: int = 0
        self.interpretability: Interpretability = interpretability

    def prepare_prompt(self, chat: Chat, resume_gen=False, model_role=None) -> str:
        """
        Prepares the prompt to include the current part of the sample.

        :param model_role: role of the model in the conversation
        :param resume_gen: whether to resume the generation
        :param chat: the current chat
        :return: prompt with the task and the current part
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

    def apply_setting(self, decoded_output: str, chat: Chat = None) -> tuple:
        """
        Postprocesses the output of the model.
        For the baseline model, this postprocessing just parses the output.

        :param decoded_output: the current output of the student
        :param chat: the current chat, only necessary in the SD and feedback setting
        :return: parsed output
        """
        return utils.parse_output(output=decoded_output)
