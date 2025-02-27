from __future__ import annotations

from prompts.Prompt import Prompt
from settings.Model import Model
from settings.baseline.Baseline import Baseline
from settings.config import Enumerate


class Skyline(Baseline):
    """
    The skyline class.
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
        Class Skyline manages the experiment with the big model. It is a subclass of Baseline, as the needs are similar.

        :param model: the model to use
        :param to_enumerate: dictionary with the settings to enumerate
        :param total_tasks: total number of tasks
        :param total_parts: total number of parts
        :param samples_per_task: number of samples per task for logging
        :param prompt: system prompt to start conversations
        """

        super().__init__(
            model,
            to_enumerate,
            total_tasks,
            total_parts,
            samples_per_task,
            prompt,
        )
