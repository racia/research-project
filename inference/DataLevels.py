from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from prettytable import HRuleStyle

from evaluation.Evaluator import AnswerEvaluator
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

    attrs = [
        "there",
        "verbs",
        "pronouns",
        "not_mentioned",
        "context_sents_hall",
    ]

    def __init__(
        self,
        there: int,
        verbs: int,
        pronouns: int,
        not_mentioned: int,
        context_sents_hall: int,
        version: str,
    ):
        self.there: int = there
        self.verbs: int = verbs
        self.pronouns: int = pronouns
        self.not_mentioned: int = not_mentioned
        self.context_sents_hall: int = context_sents_hall

        self.version: str = version

    def __add__(self, other: Features) -> Features:
        """
        Add the features of two parts.

        :param other: the other part
        :return: the sum of the features in a new object
        """
        if self.version != other.version:
            raise ValueError(
                f"Cannot add features with different versions: {self.version} and {other.version}"
            )
        return Features(
            **{attr: getattr(self, attr) + getattr(other, attr) for attr in self.attrs},
            version=self.version,
        )

    def __iadd__(self, other: Features) -> Features:
        """
        Add the features of two parts in place.

        :param other: the other part
        :return: the sum of the features in the current object
        """
        for attr in self.attrs:
            setattr(self, attr, getattr(self, attr) + getattr(other, attr))
        return self

    def get(self) -> dict[str, int]:
        """
        Get the features as a dictionary.

        :return: the features as a dictionary
        """
        return {
            f"{attr}_{self.version}": getattr(self, attr)
            for attr in self.attrs
            if hasattr(self, attr)
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

    result_attrs: list[str] = [
        "model_answer",
        "answer_correct",
        "model_reasoning",
        "reasoning_correct",
        "model_output",
        "max_supp_target",
    ]

    def __init__(
        self,
        model_output: str,
        model_answer: str,
        model_reasoning: str,
        answer_correct: bool = None,
        reasoning_correct: bool = None,
        interpretability: InterResult = None,
        version: str = "after",
    ):
        """
        Initialize the Results class.

        :param model_output: the output of the model
        :param model_answer: the answer to the question
        :param model_reasoning: the reasoning for the answer
        :param answer_correct: whether the answer is correct
        :param reasoning_correct: whether the reasoning is correct
        :param interpretability: the result of interpretability
        :param version: "after" if the setting was already applied to the model's output else "before"
        """
        if version not in ["before", "after"]:
            raise ValueError("Version should be either 'before' or 'after'.")

        self.version = version

        self.model_output: str = model_output
        self.model_answer: str = model_answer
        self.model_reasoning: str = model_reasoning

        self.answer_correct: bool = answer_correct
        self.reasoning_correct: bool = reasoning_correct

        self.features: Features = self.inspect_answer()

        self.interpretability: InterResult = interpretability
        self.max_supp_target: float = (
            interpretability.max_supp_target if interpretability else None
        )

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
            not_mentioned=contains_not_mentioned(self.model_output),
            context_sents_hall=context_sentences(self.model_output),
            version=self.version,
        )
        return self.features

    def get_result(self) -> dict[str, int | str]:
        """
        Get the result of the part.
        :return: the result of the part
        """
        try:
            attributes = {
                f"{attr}_{self.version}": getattr(self, attr)
                for attr in self.result_attrs
                if hasattr(self, attr)
            }
        except AttributeError as error:
            print(f"Error accessing attribute: {error}")
            attributes = {}
        return {**attributes, **self.features.get()}


class SuppFactInSelf:
    fully: str = "fully"
    partially: str = "partially"
    none: str = "none"


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
        "answer_lies_in_self",
        "feedback_iterations",
    ]
    current_iteration_count = 0

    def __init__(
        self,
        id_: int,
        task_id: int,
        sample_id: int,
        part_id: int,
        golden_answer: str,
        silver_reasoning: str = None,
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
            self.raw: dict = raw

            self.wrapper: Wrapper = (
                wrapper
                if wrapper
                else Wrapper(context="", question="", reasoning="", answer="")
            )
            self.to_enumerate: Enumerate = (
                to_enumerate
                if to_enumerate
                else Enumerate(context=True, question=False)
            )

            self.supporting_sent_inx: list[int] = raw.get("supporting_fact", [])
            self.answer_lies_in_self: str = self.contains_supp_sentences()

            self.structured_context, self.structured_question = structure_part(
                self.raw, self.to_enumerate
            )
            self.unwrapped_task: str = "\n".join(
                (self.structured_context, self.structured_question)
            )

            self.task: str = "\n".join(self.wrap_part()).strip()

        elif task:
            self.task: str = task

        self.golden_answer: str = golden_answer
        self.silver_reasoning: str = silver_reasoning

        self.result_before: Results = Results(
            model_output="",
            model_answer="",
            model_reasoning="",
            interpretability=None,
            version="before",
        )
        self.result_after: Results = Results(
            model_output="",
            model_answer="",
            model_reasoning="",
            interpretability=None,
            version="after",
        )
        self.feedback_iterations: int = 0

    def contains_supp_sentences(self):
        part_line_nums = [int(line_num) for line_num in self.raw["context"].keys()]
        supp_sents_in_self = set(part_line_nums).intersection(
            set(self.supporting_sent_inx)
        )
        if supp_sents_in_self:
            if len(supp_sents_in_self) == len(self.supporting_sent_inx):
                return SuppFactInSelf.fully
            elif len(supp_sents_in_self) < len(self.supporting_sent_inx):
                return SuppFactInSelf.partially
            else:
                raise ValueError(
                    f"The part has more supporting sentences than noted in the data: {self}."
                )
        return SuppFactInSelf.none

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
                return (
                    wrapped_attr.format(**replacements)
                    if wrapped_attr
                    else "\n".join(replacements.values())
                )
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
        return (
            f"<SamplePart: id_={self.id_}, task_id={self.task_id}, sample_id={self.sample_id}, "
            f"part_id={self.part_id}>"
        )

    def set_output(
        self,
        model_output: str,
        model_answer: str,
        model_reasoning: str,
        interpretability: InterResult | None,
        version: str = "after",
        feedback_iterations: int = 0,
    ) -> None:
        """
        Set the output of the model.

        :param model_output: the output of the model
        :param model_answer: the answer to the question
        :param model_reasoning: the reasoning for the answer
        :param interpretability: the interpretability result
        :param version: "after" if the setting was already applied to the model's output else "before"
        :param feedback_iterations: the number of feedback iterations

        :return: None
        """
        if type(model_answer) is not str:
            model_answer = str(model_answer)

        if not interpretability:
            interpretability = InterResult(
                attn_scores=np.ndarray([]),
                x_tokens=[],
                y_tokens=[],
                max_supp_target=0.0,
            )
        # TODO: add the score for reasoning
        if version == "after":
            self.result_after = Results(
                model_output=model_output,
                model_answer=model_answer,
                model_reasoning=model_reasoning,
                answer_correct=stats.are_identical(model_answer, self.golden_answer),
                interpretability=interpretability,
                version=version,
            )
            self.feedback_iterations = (
                feedback_iterations
                if feedback_iterations
                else SamplePart.current_iteration_count
            )
        elif version == "before":
            self.result_before = Results(
                model_output=model_output,
                model_answer=model_answer,
                model_reasoning=model_reasoning,
                answer_correct=stats.are_identical(model_answer, self.golden_answer),
                interpretability=interpretability,
                version=version,
            )
        else:
            raise ValueError(
                f"Version should be either 'before' or 'after', currently: {version}"
            )

    def get_result(self) -> dict[str, int | str]:
        """
        Get the result of the part.

        :return: the result of the part
        """
        try:
            attributes = {
                attr: getattr(self, attr)
                for attr in self.result_attrs
                if hasattr(self, attr)
            }
        except AttributeError as error:
            print(f"Error accessing attribute: {error}")
            attributes = {}

        if self.multi_system:
            return {**attributes, **self.result_before.dict, **self.result_after.dict}
        attributes.pop("feedback_iterations", None)
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
            level="sample", version="before"
        )
        self.evaluator_after: AnswerEvaluator = AnswerEvaluator(
            level="sample", version="after"
        )

        self.features_before: Features = Features(0, 0, 0, 0, 0, version="before")
        self.features_after: Features = Features(0, 0, 0, 0, 0, version="after")

        self.results = []

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
        elif type(silver_reasoning) is list:
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
            # if part.result_before.interpretability:
            #     self.evaluator_before.max_supp_target.add(
            #         part.result_before.interpretability.max_supp_target
            #     )
        self.evaluator_after.pred_answers.append(part.result_after.model_answer)
        self.evaluator_after.pred_reasonings.append(part.result_after.model_reasoning)
        # if part.result_after.interpretability:
        #     self.evaluator_after.max_supp_target.add(
        #         part.result_after.interpretability.max_supp_target
        #     )

    def print_sample_predictions(self) -> None:
        """
        Print the model's predictions to compare with true values as a table.

        :return: None
        """
        if self.multi_system:
            table = PrettyTable()
            table.field_names = [
                "GOLDEN",
                "PREDICTED-Bef",
                "PREDICTED-Aft",
                "REASONING-Bef",
                "REASONING-Aft",
            ]

            attributes = zip(
                self.evaluator_after.golden_answers,
                self.evaluator_before.pred_answers,
                self.evaluator_after.pred_answers,
                self.evaluator_before.pred_reasonings,
                self.evaluator_after.pred_reasonings,
            )

            for golden, pred_bef, pred_aft, reas_bef, reas_aft in attributes:
                table.add_row(
                    [
                        golden,
                        pred_bef.replace("\n", " "),
                        pred_aft.replace("\n", " "),
                        wrap_text(reas_bef),
                        wrap_text(reas_aft),
                    ]
                )

        else:
            table = PrettyTable()
            table.field_names = ["GOLDEN", "PREDICTED-Aft", "REASONING-Aft"]

            attributes = zip(
                self.evaluator_after.golden_answers,
                self.evaluator_after.pred_answers,
                self.evaluator_after.pred_reasonings,
            )

            for golden, pred_aft, reas_aft in attributes:
                table.add_row(
                    [golden, pred_aft.replace("\n", " "), wrap_text(reas_aft)]
                )

        table.hrules = HRuleStyle.ALL
        table.padding_width = 2  # Adds more space inside each cell

        print(f"Model's predictions for the sample {self.sample_id}:\n")
        print(table)

    def set_results(self) -> None:
        """
        Set the results for the sample by combining the results from its parts.
        :return: None
        """
        self.results = [part.get_result() for part in self.parts]
        if self.multi_system:
            self.features_before = sum(
                [part.result_before.features for part in self.parts],
                self.features_before,
            )
        self.features_after = sum(
            [part.result_after.features for part in self.parts], self.features_after
        )

    def calculate_metrics(self):
        if self.multi_system:
            self.evaluator_before.calculate_accuracies()
        self.evaluator_after.calculate_accuracies()


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
            level="task", version="before"
        )
        self.evaluator_after: MetricEvaluator = MetricEvaluator(
            level="task", version="after"
        )

        self.features_before: Features = Features(0, 0, 0, 0, 0, version="before")
        self.features_after: Features = Features(0, 0, 0, 0, 0, version="after")

        self.results: list[dict[str, str | int | float]] = []
        self.accuracies: dict = {}

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

        # self.results[0]["max_supp_target"] = self.evaluator_after.max_supp_target

    def calculate_metrics(self) -> None:
        """
        Calculate the metrics for the task.
        :return: None
        """
        accuracies = {
            "task_id": self.task_id,
        }
        # it is important to KEEP ORDER of the accuracies: before, then after
        if self.multi_system:
            self.evaluator_before.calculate_std()
            accuracies.update(**self.evaluator_before.get_accuracies())
        self.evaluator_after.calculate_std()
        accuracies.update(**self.evaluator_after.get_accuracies())
        self.accuracies = accuracies


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

        self.features_before: Features = Features(0, 0, 0, 0, 0, version="before")
        self.features_after: Features = Features(0, 0, 0, 0, 0, version="after")

        self.evaluator_before: MetricEvaluator = MetricEvaluator(
            level="split", version="before"
        )
        self.evaluator_after: MetricEvaluator = MetricEvaluator(
            level="split", version="after"
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


def print_metrics(data_level: Sample | Task | Split, table: bool = True) -> None:
    """
    Print the metrics from the evaluator in a pretty table or with the usual statements.

    :param data_level: the data level to print the metrics for
    :param table: if to print the metrics in a table
    :return: None
    """
    id_ = None
    if type(data_level) is Sample:
        id_ = data_level.sample_id
    elif type(data_level) is Task:
        id_ = data_level.task_id
    elif type(data_level) is Split:
        id_ = data_level.name

    if table:
        print_metrics_table(
            eval_before=(
                data_level.evaluator_before if data_level.multi_system else None
            ),
            eval_after=data_level.evaluator_after,
            id_=id_,
        )
    else:
        if data_level.multi_system:
            data_level.evaluator_before.print_accuracies(id_=id_)
        data_level.evaluator_after.print_accuracies(id_=id_)
