from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.Evaluator import AnswerEvaluator, MetricEvaluator
from evaluation.Statistics import Statistics
from inference.utils import *
from interpretability.utils import InterpretabilityResult as InterResult
from settings.config import Enumerate, Wrapper
from settings.utils import structure_part

stats = Statistics()


@dataclass
class Features:
    """
    This class handles the tracking of features.
    """

    def __init__(
        self,
        there: int,
        verbs: int,
        pronouns: int,
        not_mentioned: int,
        after: bool = True,
    ):
        self.there: int = there
        self.verbs: int = verbs
        self.pronouns: int = pronouns
        self.not_mentioned: int = not_mentioned

        self.after: bool = after

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

    def get(self, after: bool = True) -> dict[str, int]:
        """
        Get the features as a dictionary.

        :return: the features as a dictionary
        """
        add = "after" if self.after else "before"
        return {
            f"there_{add}": self.there,
            f"verbs_{add}": self.verbs,
            f"pronouns_{add}": self.pronouns,
            f"not_mentioned_{add}": self.not_mentioned,
        }

    def __repr__(self) -> str:
        """
        Return the features values as a string.

        :return: None
        """
        return f"<Features: {str(self.get())}>"


class Results:
    """
    Abstract class for data levels.
    """

    def __init__(
        self,
        model_output: str,
        model_answer: str,
        model_reasoning: str,
        interpretability: InterResult = None,
        after: bool = True,
    ):
        """
        Initialize the Results class.

        :param model_output: the output of the model
        :param model_answer: the answer to the question
        :param model_reasoning: the reasoning for the answer
        :param interpretability: the result of interpretability
        """
        self.after = after

        self.model_output: str = model_output
        self.model_answer: str = model_answer
        self.model_reasoning: str = model_reasoning

        self.answer_correct: bool = None
        self.reasoning_correct: bool = None

        self.features: Features = self.inspect_answer()

        self.interpretability: InterResult = interpretability

        self.result_attrs: list[str] = [
            "model_answer",
            "correct",
            "model_reasoning",
            "model_output",
        ]

        self.dict: dict = self.get_result()

    def __repr__(self):
        return f"<Results: {self.dict}>"

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
            after=self.after,
        )
        return self.features

    def get_result(self) -> dict[str, int | str]:
        """
        Get the result of the part.

        :return: the result of the part
        """
        add = "after" if self.after else "before"
        try:
            attributes = {
                f"{attr}_{add}": getattr(self, attr)
                for attr in self.result_attrs
                if hasattr(self, attr)
            }
        except AttributeError as error:
            print(f"Error accessing attribute: {error}")
            attributes = {}
        return {**attributes, **self.features.get(self.after)}


class SamplePart:
    """
    This class handles the parts of the samples, dividing it by questions.
    """

    result_attrs: list[str] = [
        "id_",
        "task_id",
        "sample_id",
        "part_id",
        "task",
        "golden_answer",
        "silver_reasoning",
    ]

    def __init__(
        self,
        id_: int,
        task_id: int,
        sample_id: int,
        part_id: int,
        golden_answer: str,
        silver_reasoning=None,
        raw: dict = None,
        task: str = None,
        wrapper: Wrapper = None,
        to_enumerate: Enumerate = None,
        multi_system: bool = False,
    ):
        """
        Initialize the part.

        :param id_: the id of the part
        :param task_id: the id of the task
        :param sample_id: the id of the sample
        :param part_id: the id of the part
        :param golden_answer: the golden answer
        :param silver_reasoning: the silver reasoning
        :param raw: the raw data of the part to format it into a task (used only for inference)
        :param task: the task for the model (used only for evaluation)
        :param wrapper: the wrapper for the task
        :param to_enumerate: if to enumerate the context sentences and the question
        :param multi_system: whether the part is for the setting with two models
        """
        self.id_: int = id_
        self.task_id: int = task_id
        self.sample_id: int = sample_id
        self.part_id: int = part_id

        self.multi_system = multi_system

        if raw and task or not (raw or task):
            raise ValueError(
                "Either 'raw' or 'task' should be provided, not both or neither."
            )

        if raw:
            if not (wrapper and to_enumerate):
                raise ValueError(
                    "'Wrapper' and 'to_enumerate' should be provided when creating tasks from scratch."
                )

            self.raw: dict = raw

            self.wrapper: Wrapper = wrapper
            self.to_enumerate: Enumerate = to_enumerate
            self.supporting_sent_inx = raw.get("supporting_fact", [])

            self.structured_context, self.structured_question = structure_part(
                self.raw, self.to_enumerate
            )
            self.unwrapped_task: str = "\n".join(
                (self.structured_context, self.structured_question)
            ).strip()

            wrapped_context, wrapped_question, reasoning_wrapper, answer_wrapper = (
                self.wrap_part()
            )
            self.task: str = "\n".join(
                (wrapped_context, wrapped_question, reasoning_wrapper, answer_wrapper)
            ).strip()

        elif task:
            self.task: str = task

        self.golden_answer: str = golden_answer
        self.silver_reasoning: str = silver_reasoning

        self.result_before: Results = Results(
            model_output="",
            model_answer="",
            model_reasoning="",
            interpretability=InterResult(
                attn_scores=np.ndarray(0), x_tokens=[], y_tokens=[]
            ),
            after=False,
        )
        self.result_after: Results = Results(
            model_output="",
            model_answer="",
            model_reasoning="",
            interpretability=InterResult(
                attn_scores=np.ndarray(0), x_tokens=[], y_tokens=[]
            ),
            after=True,
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

    def wrap_part(self) -> tuple[str, ...] | tuple[str, str, str, str]:
        """
        Format the prompt part with the wrapper.

        :return: the formatted prompt part
        """
        replacements = {
            "context": self.structured_context,
            "question": self.structured_question,
            "reasoning": "",
            "answer": "",
        }
        if self.wrapper:
            return tuple(self.wrap(attr, replacements) for attr in replacements.keys())
        return self.structured_context, self.structured_question, "", ""

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
        return f"<SamplePart: id_={self.id_}, task_id={self.task_id}, sample_id={self.sample_id}, part_id={self.part_id}>"

    def set_output(
        self,
        model_output: str,
        answer: str,
        reasoning: str,
        interpretability: InterResult,
        after: bool = True,
    ) -> None:
        """
        Set the output of the model.

        :param model_output: the output of the model
        :param answer: the answer to the question
        :param reasoning: the reasoning for the answer
        :param interpretability: the interpretability result
        :param after: whether the output is after the setting was applied to the model's output

        :return: None
        """
        if after:
            self.result_after = Results(
                model_output=model_output,
                model_answer=answer,
                model_reasoning=reasoning,
                interpretability=interpretability,
                after=after,
            )
            self.result_after.answer_correct = stats.are_identical(
                answer, self.golden_answer
            )
            # TODO: add the score for reasoning
        else:
            self.result_before = Results(
                model_output=model_output,
                model_answer=answer,
                model_reasoning=reasoning,
                interpretability=interpretability,
                after=False,
            )
            self.result_before.answer_correct = stats.are_identical(
                answer, self.golden_answer
            )

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

        if self.multi_system:
            return {**attributes, **self.result_before.dict, **self.result_after.dict}
        return {**attributes, **self.result_after.dict}


class Sample:
    """
    This class handles the samples.
    """

    def __init__(
        self,
        task_id: int,
        sample_id: int,
        multi_system: bool = False,
    ):
        self.task_id = task_id
        self.sample_id: int = sample_id

        self.multi_system: bool = multi_system

        self.parts: list[SamplePart] = []

        self.evaluator_before: AnswerEvaluator = AnswerEvaluator(
            level="sample", after=False
        )
        self.evaluator_after: AnswerEvaluator = AnswerEvaluator(
            level="sample", after=True
        )

    def add_golden_answers(self, golden_answer: str | list[str]) -> None:
        """
        Add the golden answers to the sample.

        :param golden_answer: the golden answer(s) to add
        :return: None
        """
        if type(golden_answer) is str:
            self.evaluator_before.golden_answers.append(golden_answer)
            self.evaluator_after.golden_answers.append(golden_answer)
        else:
            self.evaluator_before.golden_answers.extend(golden_answer)
            self.evaluator_after.golden_answers.extend(golden_answer)

    def add_silver_reasoning(self, silver_reasoning: str | list[str]) -> None:
        """
        Add the silver reasoning to the sample.

        :param silver_reasoning: the silver reasoning(s) to add
        :return: None
        """
        if type(silver_reasoning) is str:
            self.evaluator_before.silver_reasonings.append(silver_reasoning)
            self.evaluator_after.silver_reasonings.append(silver_reasoning)
        else:
            self.evaluator_before.silver_reasonings.extend(silver_reasoning)
            self.evaluator_after.silver_reasonings.extend(silver_reasoning)

    def add_part(self, part: SamplePart) -> None:
        """
        Add a part to the sample.
        Should be called after the output of the part is set with SamplePart.set_output().

        :param part: the part to add
        :return: None
        """
        self.parts.append(part)

        if self.multi_system:
            self.evaluator_before.pred_answers.append(part.result_before.model_answer)
            self.evaluator_before.pred_reasonings.append(
                part.result_before.model_reasoning
            )
        self.evaluator_after.pred_answers.append(part.result_after.model_answer)
        self.evaluator_after.pred_reasonings.append(part.result_after.model_reasoning)

    def print_sample_predictions(self) -> None:
        """
        Print the model's predictions to compare with true values.

        :return: None
        """
        if self.multi_system:
            attributes = zip(
                self.evaluator_after.golden_answers,
                self.evaluator_before.pred_answers,
                self.evaluator_after.pred_answers,
                self.evaluator_before.pred_reasonings,
                self.evaluator_after.pred_reasonings,
            )
            print(
                f"Model's predictions for the sample {self.sample_id}:",
                "\t{0:<18s} PREDICTED-{1:<18s} PREDICTED-{2:<18s} REASONING-{1:<36s} REASONING-{2:<36s}".format(
                    "GOLDEN", "Bef", "Aft"
                ),
                sep="\n\n",
                end="\n\n",
            )
            for (
                golden_answer,
                pred_answer_bef,
                pred_answer_aft,
                pred_reasoning_bef,
                pred_reasoning_aft,
            ) in attributes:
                print(
                    "\t{0:<18s} {1:<18s} {2:<18s} {3:<36s} {4:<36s}".format(
                        golden_answer,
                        pred_answer_bef.replace("\n", "\t"),
                        pred_answer_aft.replace("\n", "\t"),
                        pred_reasoning_bef.replace("\n", "\t"),
                        pred_reasoning_aft.replace("\n", "\t"),
                    ),
                )
        else:
            attributes = zip(
                self.evaluator_after.golden_answers,
                self.evaluator_after.pred_answers,
                self.evaluator_after.pred_reasonings,
            )
            print(
                "Model's predictions for the sample:",
                "\t{0:<18s} {1:<25s} {2} ".format(
                    "GOLDEN", "PREDICTED-Aft", "REASONING-Aft"
                ),
                sep="\n\n",
                end="\n\n",
            )
            for (
                golden_answer,
                pred_answer_aft,
                pred_reasoning_aft,
            ) in attributes:
                print(
                    "\t{0:<18s} {1:<25s} {2}".format(
                        golden_answer,
                        pred_answer_aft.replace("\n", "\t"),
                        pred_reasoning_aft.replace("\n", "\t"),
                    ),
                )


class Task:
    """
    This class handles the tasks.
    """

    def __init__(self, task_id: int, multi_system: bool = False):
        """
        Initialize the task.

        :param task_id: the id of the task
        :param multi_system: whether the task is for the setting with two models
        """
        self.task_id = task_id
        self.multi_system = multi_system

        self.samples: list[Sample] = []
        self.parts: list[SamplePart] = []

        self.evaluator_before: MetricEvaluator = MetricEvaluator(
            level="task", after=False
        )
        self.evaluator_after: MetricEvaluator = MetricEvaluator(
            level="task", after=True
        )

        self.features_before: Features = Features(0, 0, 0, 0)
        self.features_after: Features = Features(0, 0, 0, 0)

        self.results: list[dict[str, str | int | float]] = []

    def add_sample(self, sample: Sample) -> None:
        """
        Add a sample to the task.

        :param sample: the sample to add
        :return: None
        """
        self.samples.append(sample)
        if self.multi_system:
            self.evaluator_before.update(sample.evaluator_before)
        self.evaluator_after.update(sample.evaluator_after)

    def set_results(self) -> None:
        """
        Set the results for the task by combining the results from the parts of all samples.
        Additionally, calculate the mean of the accuracy metrics and add them to the results.

        :return: None
        """
        self.parts = [part for sample in self.samples for part in sample.parts]
        self.results = [part.get_result() for part in self.parts]

        if self.multi_system:
            self.features_before = sum(
                [part.result_before.features for part in self.parts],
                self.features_before,
            )
            self.results[0][
                "exact_match_accuracy_before"
            ] = self.evaluator_before.exact_match_accuracy.get_mean()
            self.results[0][
                "soft_match_accuracy_before"
            ] = self.evaluator_before.soft_match_accuracy.get_mean()

        self.features_after = sum(
            [part.result_after.features for part in self.parts], self.features_after
        )
        self.results[0][
            "exact_match_accuracy_after"
        ] = self.evaluator_after.exact_match_accuracy.get_mean()
        self.results[0][
            "soft_match_accuracy_after"
        ] = self.evaluator_after.soft_match_accuracy.get_mean()


class Split:
    """
    This class handles the splits of the data.
    """

    def __init__(self, name: str, multi_system: bool = False):
        """
        Initialize the Split.

        :param name: the name of the split
        :param multi_system: whether the split is for the setting with two models
        """
        self.name: str = name
        self.multi_system: bool = multi_system

        self.tasks: list[Task] = []

        self.features_before: Features = Features(0, 0, 0, 0)
        self.features_after: Features = Features(0, 0, 0, 0)

        self.evaluator_before: MetricEvaluator = MetricEvaluator(
            level="split", after=False
        )
        self.evaluator_after: MetricEvaluator = MetricEvaluator(
            level="split", after=True
        )

    def add_task(self, task: Task) -> None:
        """
        Add a task to the prompt level data.

        :param task: the task to add
        :return: None
        """
        self.tasks.append(task)
        if self.multi_system:
            self.features_before += task.features_before
            self.evaluator_before.update(task.evaluator_before)
        self.features_after += task.features_after
        self.evaluator_after.update(task.evaluator_after)
