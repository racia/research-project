from __future__ import annotations

from data.DataSaver import DataSaver
from inference.Prompt import Prompt
from settings.Model import Model
from settings.baseline.Baseline import Baseline


class Skyline(Baseline):
    """
    The skyline class.
    """

    def __init__(
        self,
        model: Model,
        total_tasks: int,
        total_parts: int,
        samples_per_task: int = 5,
        init_prompt: Prompt = None,
        saver: DataSaver = None,
    ):
        """
        Class Skyline manages the experiment with the big model. It is a subclass of Baseline, as the needs are similar.

        :param model: the model to use
        :param init_prompt: system prompt to start conversations
        :param total_tasks: total number of tasks
        :param total_parts: total number of parts
        :param samples_per_task: number of samples per task for logging
        :param init_prompt: system prompt to start conversations
        :param saver: data saver to use
        """

        super().__init__(
            model=model,
            total_tasks=total_tasks,
            total_parts=total_parts,
            samples_per_task=samples_per_task,
            init_prompt=init_prompt,
            saver=saver,
        )
