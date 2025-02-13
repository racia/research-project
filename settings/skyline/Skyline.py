from __future__ import annotations

from data.Statistics import Statistics
from prompts.Prompt import Prompt
from settings.Model import Model
from settings.baseline.Baseline import Baseline
from settings.utils import Enumerate


class Skyline(Baseline):
    """
    The skyline class.
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
        Class Skyline manages the experiment with the big model. It is a subclass of Baseline, as the needs are similar.

        :param parse_output: if we want to parse the output of the model (currently looks for 'answer' and 'reasoning')
        :param statistics: class for statistics
        :param prompt: system prompt to start conversations
        :param samples_per_task: number of samples per task for logging
        """

        super().__init__(
            model, to_enumerate, parse_output, statistics, prompt, samples_per_task
        )
