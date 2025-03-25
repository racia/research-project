from __future__ import annotations

from inference.Prompt import Prompt
from interpretability.Interpretability import Interpretability
from settings.Model import Model
from settings.baseline.Baseline import Baseline
from settings.config import Enumerate, Wrapper


class Skyline(Baseline):
    """
    The skyline class.
    """

    def __init__(
        self,
        model: Model,
        to_enumerate: Enumerate,
        total_tasks: int,
        total_parts: int,
        interpretability: Interpretability,
        samples_per_task: int = 5,
        init_prompt: Prompt = None,
        wrapper: Wrapper = None,
    ):
        """
        Class Skyline manages the experiment with the big model. It is a subclass of Baseline, as the needs are similar.

        :param model: the model to use
        :param to_enumerate: dictionary with the settings to enumerate
        :param total_tasks: total number of tasks
        :param total_parts: total number of parts
        :param samples_per_task: number of samples per task for logging
        :param init_prompt: system prompt to start conversations
        :param wrapper: wrapper for the model
        :param interpretability: optional interpretability instance
        """

        super().__init__(
            model=model,
            total_tasks=total_tasks,
            total_parts=total_parts,
            samples_per_task=samples_per_task,
            init_prompt=init_prompt,
            to_enumerate=to_enumerate,
            wrapper=wrapper,
            interpretability=interpretability,
            mode=mode
        )
