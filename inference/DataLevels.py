from __future__ import annotations

from dataclasses import dataclass

from numpy import ndarray

from evaluation.Evaluator import AnswerEvaluator, MetricEvaluator
from evaluation.Statistics import Statistics
from inference.utils import *
from interpretability.Interpretability import InterpretabilityResult as InterResult
from settings.config import Enumerate, Wrapper
from settings.utils import structure_part


@dataclass
class Features:
    """
    This class handles the tracking of features.
    """

    there: int
    verbs: int
    pronouns: int
    not_mentioned: int

    def __add__(self, other: Features) -> Features:
        """
        Add the features of two parts.

        :param other: the other part
        :return: the sum of the features in a new object
        """
        return Features(
            there=self.there + other.there,
            verbs=self.verbs + other.verbs,
            pronouns=self.pronouns + other.pronouns,
            not_mentioned=self.not_mentioned + other.not_mentioned,
        )

    def __iadd__(self, other: Features) -> Features:
        """
        Add the features of two parts in place.

        :param other: the other part
        :return: the sum of the features in the current object
        """
        self.there += other.there
        self.verbs += other.verbs
        self.pronouns += other.pronouns
        self.not_mentioned += other.not_mentioned
        return self

    def get(self) -> dict[str, int]:
        """
        Get the features as a dictionary.

        :return: the features as a dictionary
        """
        return {
            "there": self.there,
            "verbs": self.verbs,
            "pronouns": self.pronouns,
            "not_mentioned": self.not_mentioned,
        }

    def __repr__(self) -> str:
        """
        Return the features values as a string.

        :return: None
        """
        return f"<Features: {str(self.get())}>"


class DataLevel:
    """
    Abstract class for data levels.
    """

    def __init__(self, task_id: int):
        self.task_id = task_id
        self.features = Features(
            there=0,
            verbs=0,
            pronouns=0,
            not_mentioned=0,
        )


class SamplePart(DataLevel):
    """
    This class handles the parts of the samples, dividing it by questions.
    """

    result_attrs = [
        "id_",
        "task_id",
        "sample_id",
        "part_id",
        "task",
        "model_reasoning",
        "model_answer",
        "correct",
        "golden_answer",
        "silver_reasoning",
        "model_output",
    ]

    def __init__(
        self,
        id_: int,
        task_id: int,
        sample_id: int,
        part_id: int,
        raw: dict,
        golden_answer: str,
        silver_reasoning=None,
        wrapper: Wrapper = None,
        to_enumerate: Enumerate = None,
    ):
        super().__init__(task_id)
        self.id_ = id_
        self.sample_id = sample_id
        self.part_id = part_id

        self.raw = raw
        self.wrapper = wrapper
        self.to_enumerate = to_enumerate

        context, question, pre_reasoning, pre_answer = self.format_part()

        self.context = context.split("\n")
        self.question = question

        self.task = "\n".join((context, question, pre_reasoning, pre_answer)).strip()

        self.golden_answer = golden_answer
        self.silver_reasoning = silver_reasoning

        self.model_output = None
        self.model_reasoning = None
        self.model_answer = None

        self.correct = None

        self.stats = Statistics()
        self.features = Features(
            there=0,
            verbs=0,
            pronouns=0,
            not_mentioned=0,
        )
        self.result = None
        self.interpretability: InterResult = InterResult(
            attn_scores=ndarray([]),
            x_tokens=[],
            y_tokens=[],
        )

    def wrap(self, attr: str, replacements: dict[str, str]) -> str:
        """
        Wrap the attribute with the wrapper, allowing flexible placeholders.

        :param attr: The name of the attribute to wrap (e.g., 'context', 'question')
        :param replacements: A dictionary of placeholders to replace in the template
        :return: The formatted attribute if it exists, otherwise None
        """
        if hasattr(self.wrapper, attr):
            wrapped_attr = getattr(self.wrapper, attr)
            try:
                return wrapped_attr.format(**replacements) if wrapped_attr else ""
            except KeyError as e:
                raise ValueError(f"Missing placeholder in replacements: {e}")
        else:
            raise AttributeError(f"Wrapper has no attribute '{attr}': {self.wrapper}")

    def format_part(self) -> tuple[str, ...] | tuple[str, str, str, str]:
        """
        Format the prompt part with the wrapper.

        :return: the formatted prompt part
        """
        context, question = structure_part(self.raw, self.to_enumerate)
        replacements = {
            "context": context,
            "question": question,
            "reasoning": "",
            "answer": "",
        }
        if self.wrapper:
            return tuple(self.wrap(attr, replacements) for attr in replacements.keys())
        return context, question, "", ""

    def __str__(self) -> str:
        """
        Return the task which includes the wrapped context, question,
        reasoning, and answer as the task for the model (no actual reasoning and answer).

        :return: the task for the model
        """
        return self.task

    def __repr__(self) -> str:
        """
        Return the string representation of the part.

        :return: the string representation of the part
        """
        return f"<SamplePart: id={self.id_}, task_id={self.task_id}, sample_id={self.sample_id}, part_id={self.part_id}>"

    def inspect_answer(self) -> Features:
        """
        Evaluate the answer by checking if the answer is correct and contains
        - 'there',
        - a verb,
        - 'not mentioned',
        - pronouns (instead of names).

        :return: Features object with the results
        """
        self.features = Features(
            there=contains_there(self.model_answer),
            verbs=contains_verb(self.model_answer),
            pronouns=contains_pronouns(self.model_answer),
            not_mentioned=contains_not_mentioned(self.model_answer),
        )
        return self.features

    def set_output(self, model_output: str, answer: str, reasoning: str) -> None:
        """
        Set the output of the model.

        :param model_output: the output of the model
        :param answer: the answer to the question
        :param reasoning: the reasoning for the answer

        :return: None
        """
        self.model_output = model_output
        self.model_answer = answer
        self.correct = self.stats.are_identical(self.model_answer, self.golden_answer)

        # TODO: add the score for reasoning
        self.model_reasoning = reasoning

        self.features = self.inspect_answer()
        self.result = self.get_result()

    def get_result(self) -> dict[str, int | str]:
        """
        Get the result of the part.

        :return: the result of the part
        """
        try:
            attributes = {
                attr.strip("_"): getattr(self, attr)
                for attr in self.result_attrs
                if hasattr(self, attr)
            }
        except AttributeError as error:
            print(f"Error accessing attribute: {error}")
            attributes = {}
        return {**attributes, **self.features.get()}


class Sample(DataLevel):
    """
    This class handles the samples.
    """

    def __init__(
        self,
        task_id: int,
        sample_id: int,
        evaluator: AnswerEvaluator,
    ):
        super().__init__(task_id)
        self.sample_id = sample_id

        self.parts: list[SamplePart] = []

        self.evaluator = evaluator

    def add_part(self, part: SamplePart) -> None:
        """
        Add a part to the sample.
        Should be called after the output of the part is set with SamplePart.set_output().

        :param part: the part to add
        :return: None
        """
        self.parts.append(part)
        self.features += part.features

        self.evaluator.pred_answers.append(part.model_answer)
        self.evaluator.pred_reasonings.append(part.model_reasoning)

    def print_sample_predictions(self) -> None:
        """
        Print the model's predictions to compare with true values.

        :return: None
        """
        attributes = zip(
            self.evaluator.golden_answers,
            self.evaluator.pred_answers,
            self.evaluator.pred_reasonings,
        )
        print(
            "Model's predictions for the sample:",
            "\t{0:<18s} {1:<18s} REASONING".format("GOLDEN", "PREDICTED"),
            sep="\n\n",
            end="\n\n",
        )
        for golden_answer, pred_answer, pred_reasoning in attributes:
            print(
                "\t{0:<18s} {1:<18s} {2}".format(
                    golden_answer, pred_answer.replace("\n", "\t"), pred_reasoning
                ),
            )


class Task(DataLevel):
    """
    This class handles the tasks.
    """

    def __init__(self, task_id: int, evaluator: MetricEvaluator):
        super().__init__(task_id)

        self.samples: list[Sample] = []
        self.parts: list[SamplePart] = []

        self.evaluator = evaluator
        self.results: list[dict[str, int | str]] = []

    def add_sample(self, sample: Sample) -> None:
        """
        Add a sample to the task.

        :param sample: the sample to add
        :return: None
        """
        self.samples.append(sample)
        self.features += sample.features
        self.evaluator.update(sample.evaluator)

    def set_results(self) -> None:
        """
        Set the results for the task by combining the results from the parts of all samples.
        Additionally, calculate the mean of the accuracy metrics and add them to the results.

        :return: None
        """
        self.parts = [part for sample in self.samples for part in sample.parts]
        self.results = [part.result for part in self.parts]

        self.results[0][
            "exact_match_accuracy"
        ] = self.evaluator.exact_match_accuracy.get_mean()
        self.results[0][
            "soft_match_accuracy"
        ] = self.evaluator.soft_match_accuracy.get_mean()


class Split(DataLevel):
    """
    This class handles the splits of the data.
    """

    def __init__(self, name: str, evaluator: MetricEvaluator):
        super().__init__(task_id=0)
        self.name = name
        self.tasks: list[Task] = []

        self.evaluator = evaluator

    def add_task(self, task: Task) -> None:
        """
        Add a task to the prompt level data.

        :param task: the task to add
        :return: None
        """
        self.tasks.append(task)
        self.features += task.features
        self.evaluator.update(task.evaluator)
