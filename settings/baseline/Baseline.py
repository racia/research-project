from __future__ import annotations

import settings.baseline.utils as utils
from data.Statistics import Statistics
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
        parse_output: bool,
        statistics: Statistics,
        prompt: Prompt = None,
        samples_per_task: int = 5,
    ):
        """
        Baseline class manages model runs and data flows around it.

        :param parse_output: if we want to parse the output of the model (currently looks for 'answer' and 'reasoning')
        :param statistics: class for statistics
        :param prompt: system prompt to start conversations
        :param samples_per_task: number of samples per task for logging
        """
        self.model = model

        self.prompt = prompt
        self.to_enumerate = to_enumerate
        self.parse_output = parse_output

        self.stats = statistics

        self.question_id = 0
        self.total_samples = 0
        self.total_tasks = 0
        self.total_parts = 0
        self.samples_per_task = samples_per_task

        self.accuracies_per_task: list = []
        self.soft_match_accuracies_per_task: list = []

        self.accuracy: int = 0
        self.soft_match_accuracy: int = 0

    def prepare_prompt(self, sample_part: dict[str, dict], chat: Chat) -> str:
        """
        Prepares the prompt to include the current part of the sample.

        :param sample_part: the current part of the sample
        :param chat: the chat
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
        if self.parse_output:
            parsed_output = utils.parse_output(output=decoded_output)
            return parsed_output
        return {"model_answer": decoded_output}
