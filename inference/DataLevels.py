from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

import numpy as np
from prettytable import HRuleStyle, PrettyTable

from evaluation.Evaluator import AnswerEvaluator, MetricEvaluator
from evaluation.Metrics import Metric
from evaluation.Statistics import Statistics
from inference.utils import (
    contains_not_mentioned,
    contains_pronouns,
    contains_there,
    contains_verb,
    context_sentences,
    print_metrics_table,
    structure_part,
    wrap_text,
    is_nan,
    REASONING_SCORE_MAP,
    update_attributes,
    majority_vote,
    only_none,
)
from interpretability.utils import InterpretabilityResult as InterResult
from settings.config import Enumerate, Wrapper
from settings.utils import parse_output

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
                f"Cannot addition features with different versions: {self.version} and {other.version}"
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

    CASE_COUNTERS = {
        "before": {
            "ans_null_reas_null": [],
            "ans_null_reas_corr": [],
            "ans_null_reas_incorr": [],
            "ans_corr_reas_null": [],
            "ans_corr_reas_incorr": [],
            "ans_corr_reas_corr": [],
            "ans_incorr_reas_null": [],
            "ans_incorr_reas_incorr": [],
            "ans_incorr_reas_corr": [],
        },
        "after": {
            "ans_null_reas_null": [],
            "ans_null_reas_corr": [],
            "ans_null_reas_incorr": [],
            "ans_corr_reas_null": [],
            "ans_corr_reas_incorr": [],
            "ans_corr_reas_corr": [],
            "ans_incorr_reas_null": [],
            "ans_incorr_reas_incorr": [],
            "ans_incorr_reas_corr": [],
        },
    }

    result_attrs: list[str] = [
        "model_answer",
        "answer_correct",
        "answer_in_self",
        "model_reasoning",
        "reasoning_correct",
        "model_output",
        "max_supp_attn",
        "attn_on_target",
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
        self.max_supp_attn: float = (
            interpretability.max_supp_attn if interpretability else None
        )
        self.attn_on_target: float = (
            interpretability.attn_on_target if interpretability else None
        )

        self.dict: dict = self.get_result()
        self.category: str = ""

        # used only to store the ids of the parts
        self.ids: dict[str, list[int]] = {}
        self.tokens: dict[str, list[str]] = {}

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

    def categorize(self, ids: tuple[int, ...]) -> str:
        """
        Categorize the part results into types in the CASE_COUNTERS.
        Reasoning is considered to be incorrect unless it was explicitly defined as correct.

        :param ids: the ids of the part to addition to the category
        """
        if self.model_answer == "None":
            self.model_answer = ""
        if self.model_reasoning == "None":
            self.model_reasoning = ""

        no_answer_no_reasoning = not self.model_answer and not self.model_reasoning
        no_answer_reasoning = not self.model_answer and self.model_reasoning
        answer_no_reasoning = self.model_answer and not self.model_reasoning
        answer_reasoning = self.model_answer and self.model_reasoning
        ans_corr_reas_corr = self.answer_correct and self.reasoning_correct
        ans_corr_reas_incorr = self.answer_correct and not self.reasoning_correct
        ans_incorr_reas_corr = not self.answer_correct and self.reasoning_correct
        ans_incorr_reas_incorr = not self.answer_correct and not self.reasoning_correct
        if no_answer_no_reasoning:
            category = "ans_null_reas_null"
        elif no_answer_reasoning and self.reasoning_correct:
            category = "ans_null_reas_corr"
        elif no_answer_reasoning and not self.reasoning_correct:
            category = "ans_null_reas_incorr"
        elif answer_no_reasoning and self.answer_correct:
            category = "ans_corr_reas_null"
        elif answer_reasoning and ans_corr_reas_incorr:
            category = "ans_corr_reas_incorr"
        elif answer_reasoning and ans_corr_reas_corr:
            category = "ans_corr_reas_corr"
        elif answer_no_reasoning and not self.answer_correct:
            category = "ans_incorr_reas_null"
        elif answer_reasoning and ans_incorr_reas_incorr:
            category = "ans_incorr_reas_incorr"
        elif answer_reasoning and ans_incorr_reas_corr:
            category = "ans_incorr_reas_corr"
        else:
            raise ValueError("The output type is not handled: ", self)

        ids = "\t".join(map(str, ids))
        Results.CASE_COUNTERS[self.version][category].append(ids)
        self.category = category
        return category

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
        "iterations",
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
        :param raw: the raw data of the part to format it into a task
        :param wrapper: the wrapper for the task
        :param to_enumerate: if to enumerate the context sentences and the question
        :param multi_system: whether the part is for the setting with two models
        """
        self.id_: int = id_
        self.task_id: int = task_id
        self.sample_id: int = sample_id
        self.part_id: int = part_id
        self.ids: tuple[int, ...] = (
            self.id_,
            self.task_id,
            self.sample_id,
            self.part_id,
        )

        self.multi_system = multi_system

        self.version_map = {
            "before": 0,
            "after": 1,
        }
        self.versions: list[str] = (
            ["before", "after"] if self.multi_system else ["before"]
        )

        self.raw: dict = raw
        self.wrapper: Wrapper = self.set_wrapper(wrapper)
        self.to_enumerate: Enumerate = self.set_to_enumerate(to_enumerate)

        self.supporting_sent_inx: list[int] = raw.get("supporting_facts", [])
        self.context_line_nums: list[int] = [
            int(line_num) for line_num in self.raw["context"].keys()
        ]
        self.answer_lies_in_self: str = self.contains_supp_sentences()

        self.structured_context: str = ""
        self.structured_question: str = ""
        self.unwrapped_task: str = ""
        self.task: str = self.prepare_task()
        # self.wrapped_task is only used when loading the results
        self.wrapped_task = ""

        self.golden_answer: str = golden_answer
        self.silver_reasoning: str = silver_reasoning

        self.results = [[], []]
        self.iterations: int = 0

    def set_wrapper(self, wrapper: Wrapper = None) -> Wrapper:
        """
        Set the wrapper for the task.

        :param wrapper: the wrapper for the task
        :return: None
        """
        if not wrapper:
            self.wrapper = Wrapper(context="", question="", reasoning="", answer="")
        else:
            self.wrapper = wrapper
        return self.wrapper

    def set_to_enumerate(self, to_enumerate: Enumerate = None) -> Enumerate:
        """
        Set the to_enumerate for the task.

        :param to_enumerate: the to_enumerate for the task
        :return: None
        """
        if not to_enumerate:
            self.to_enumerate = Enumerate(context=True, question=False)
        else:
            self.to_enumerate = to_enumerate
        return self.to_enumerate

    def prepare_task(self) -> str:
        """
        Prepare the task for the model.
        :return: the task for the model
        """
        if not hasattr(self, "task"):
            self.structured_context, self.structured_question = structure_part(
                self.raw, self.to_enumerate
            )
            self.unwrapped_task: str = "\n".join(
                (self.structured_context, self.structured_question)
            )
            self.task = "\n".join(self.wrap_part()).strip()
        else:
            raise ValueError(
                f"The task is already prepared: {self.task}. "
                f"Please check the previous call of this function."
            )
        return self.task

    def contains_supp_sentences(self):
        """
        Check if the supporting sentences are in the part.
        :return: the type of supporting sentences in the part
        """
        supp_sents_in_self = set(self.context_line_nums).intersection(
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

    def wrap(self, attr: str, replacement: str) -> str:
        """
        Wrap the attribute with the wrapper, allowing flexible placeholders.

        :param attr: The name of the attribute to wrap (e.g., 'context', 'question')
        :param replacement: A dictionary of placeholders to replace in the template
        :return: The formatted attribute if it exists, otherwise None
        """
        if hasattr(self.wrapper, attr):
            wrapped_attr = getattr(self.wrapper, attr)
            try:
                return (
                    wrapped_attr.format(**{attr: replacement})
                    if wrapped_attr
                    else replacement
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
            return tuple(
                self.wrap(*replacement) for replacement in replacements.items()
            )
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
        version: str,
        interpretability: InterResult | None,
        messages: list[dict] = None,
        model_output: str = None,
        model_answer: str = None,
        model_reasoning: str = None,
        wrapped_task: str = None,
        iterations: int = 0,
    ) -> None:
        """
        Set the output of the model either from the message or from the model's output.

        :param messages: the message from the model
        :param model_output: the output of the model
        :param model_answer: the answer to the question
        :param model_reasoning: the reasoning for the answer
        :param interpretability: the interpretability result
        :param version: "after" if the setting was already applied to the model's output else "before"
        :param wrapped_task: the full task for the model (when loading results)
        :param iterations: the number of iterations (feedback or speculative decoding)

        :return: None
        """
        nothing = not any([messages, model_output, model_answer, model_reasoning])
        everything = all([messages, model_output, model_answer, model_reasoning])
        if nothing or everything:
            raise ValueError(
                "Either a message or model_output, model_answer, and model_reasoning should be provided."
            )
        if not messages and not any(
            [model_output and model_answer and model_reasoning]
        ):
            raise ValueError(
                "When no message, a model_output, model_answer, and model_reasoning should be provided."
            )
        if messages:
            if len(messages) < 2:
                raise ValueError(
                    "The messages should contain at least two elements: the input and the output."
                )
            model_output = messages[-1]["content"]
            model_answer, model_reasoning = parse_output(output=model_output)

        if type(model_answer) is not str:
            model_answer = str(model_answer)

        self.wrapped_task = wrapped_task if wrapped_task is not None else self.task

        if not interpretability:
            warnings.warn(
                f"DEBUG: NO INTERPRETABILITY SCORES WERE CALCULATED "
                f"for task {self.task_id}, sample {self.sample_id}, part {self.part_id}."
            )
            interpretability = InterResult(np.ndarray([]), [], [], 0.0, 0.0)

        if version not in self.version_map:
            raise ValueError(
                f"Version should be either 'before' or 'after', currently: {version}"
            )

        result = Results(
            model_output=model_output,
            model_answer=model_answer,
            model_reasoning=model_reasoning,
            answer_correct=stats.are_identical(model_answer, self.golden_answer),
            interpretability=interpretability,
            version=version,
        )
        if messages:
            for message in messages:
                result.ids[message["role"]] = message["ids"]
                result.tokens[message["role"]] = message["tokens"]

        if version == "after":
            self.iterations = iterations or SamplePart.current_iteration_count

        result.categorize(ids=self.ids)
        self.results[self.version_map[version]].append(result)

    def get_result(self, average: bool = True) -> dict[str, int | str]:
        """
        Get the result of the part.

        :return: the result of the part
        """
        try:
            # these are not relevant for duplicated answers
            attributes = {
                attr: getattr(self, attr)
                for attr in self.result_attrs
                if hasattr(self, attr)
            }
        except AttributeError as error:
            print(f"Error accessing attribute: {error}")
            attributes = {}

        if not self.multi_system:
            attributes.pop("iterations", None)

        if not self.results:
            raise ValueError("No results found for the part.")

        # attributes: list[str] = [
        #     "id_",
        #     "task_id",
        #     "sample_id",
        #     "part_id",
        #     "task",
        #     "golden_answer",
        #     "silver_reasoning",
        #     "answer_lies_in_self",
        #     "iterations",
        # ]
        dupl_attributes = {
            attr: [] for attr in SamplePart.result_attrs + Features.attrs
        }
        for version_results in self.results:
            for i, duplicate_result in enumerate(version_results):
                dupl_attributes = update_attributes(
                    dupl_attributes, duplicate_result.dict
                )
                # result_attr: list[str] = [
                #     "model_answer",
                #     "answer_correct",
                #     "model_reasoning",
                #     "reasoning_correct",
                #     "model_output",
                #     "max_supp_attn",
                #     "attn_on_target",
                # ]
                # features_attrs = [
                #     "there",
                #     "verbs",
                #     "pronouns",
                #     "not_mentioned",
                #     "context_sents_hall",
                # ]
        averaged_attrs = {}
        for attr in dupl_attributes:
            if dupl_attributes[attr] and type(dupl_attributes[attr][0]) is str:
                averaged_attrs[attr] = "\n".join(dupl_attributes[attr])
            elif only_none(dupl_attributes[attr]):
                averaged_attrs[attr] = None
            else:
                average = sum(dupl_attributes[attr]) / len(dupl_attributes[attr])
                averaged_attrs[attr] = round(average, 4)
        return {**attributes, **averaged_attrs}

    def number_of_duplicates(self) -> int:
        """
        Get the number of duplicates for the part.

        :return: the number of duplicates for the part
        """
        if not self.results:
            raise ValueError("No results found for the part.")
        return len(self.results[0])


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
        self.versions: list[str] = (
            ["before", "after"] if self.multi_system else ["before"]
        )
        self.run_with_reasoning: bool = False
        self.parts: list[SamplePart] = []

        self.num_dupl = self.parts[0].number_of_duplicates() if self.parts else 1
        evaluator_before = [
            AnswerEvaluator(level="sample", version="before")
        ] * self.num_dupl
        self.evaluators: list[list[AnswerEvaluator]] = (
            [
                evaluator_before,
                [AnswerEvaluator(level="sample", version="after")] * self.num_dupl,
            ]
            if self.multi_system
            else [evaluator_before]
        )
        majority_eval_before = AnswerEvaluator(level="sample", version="before")
        self.majority_evaluators: list[AnswerEvaluator] = (
            [majority_eval_before, AnswerEvaluator(level="sample", version="after")]
            if self.multi_system
            else [majority_eval_before]
        )

        features_before = Features(0, 0, 0, 0, 0, version="before")
        self.features: list[Features] = (
            [features_before, Features(0, 0, 0, 0, 0, version="after")]
            if self.multi_system
            else [features_before]
        )
        self.results = []

    def add_part(self, part: SamplePart) -> None:
        """
        Add a part to the sample.
        Should be called after the output of the part is set with SamplePart.set_output().

        :param part: the part to addition
        :return: None
        """
        self.parts.append(part)
        if self.multi_system and len(part.results) != 2:
            results_before = "\n".join(part.results[0])
            results_after = "\n".join(part.results[1])
            raise ValueError(
                f"The part should have two results for the multi-system setting, "
                f"got {len(part.results)}:\n{results_before}\n{results_after}"
            )
        for i, (version_evaluators, version_results) in enumerate(
            zip(self.evaluators, part.results)
        ):
            model_answers, max_supp_attns, attns_on_target = [], [], []
            for j, (evaluator, result) in enumerate(
                zip(version_evaluators, version_results)
            ):
                answer = result.model_answer
                evaluator.golden_answers.append(part.golden_answer)
                evaluator.pred_answers.append(answer)
                model_answers.append(answer)
                evaluator.pred_reasonings.append(result.model_reasoning)
                if result.model_reasoning:
                    self.run_with_reasoning = True
                    if part.silver_reasoning:
                        evaluator.silver_reasonings.append(part.silver_reasoning)
                if result.interpretability:
                    if is_nan(result.max_supp_attn):
                        warnings.warn(
                            f"Found 'nan' in result.max_supp_attn results for part {part.ids}. Skipping."
                        )
                    else:
                        evaluator.max_supp_attn.add(result.max_supp_attn)
                        max_supp_attns.append(result.max_supp_attn)
                    if is_nan(result.attn_on_target):
                        warnings.warn(
                            f"Found 'nan' in result.attn_on_target results for part {part.ids}. Skipping."
                        )
                        evaluator.attn_on_target.add(0.0)
                    else:
                        evaluator.attn_on_target.add(result.attn_on_target)
                        attns_on_target.append(result.attn_on_target)

            majority_answer = majority_vote(model_answers)
            self.majority_evaluators[i].pred_answers.append(majority_answer)
            self.majority_evaluators[i].golden_answers.append(part.golden_answer)
            average_max_supp_attn = float(np.mean(max_supp_attns))
            average_attn_on_target = float(np.mean(attns_on_target))
            self.majority_evaluators[i].max_supp_attn.add(average_max_supp_attn)
            self.majority_evaluators[i].attn_on_target.add(average_attn_on_target)

    def print_sample_predictions(self) -> None:
        """
        Print the model's predictions to compare with true values as a table.

        :return: None
        """
        ans_bef = (
            [f"PREDICTED-Bef {i}" for i in range(1, self.num_dupl + 1)]
            if self.num_dupl > 1
            else ["PREDICTED-Bef"]
        )
        reas_bef = (
            [f"REASONING-Bef {i}" for i in range(1, self.num_dupl + 1)]
            if self.num_dupl > 1
            else ["REASONING-Bef"]
        )
        answers = [
            evaluator.pred_answers
            for version_evaluators in self.evaluators
            for evaluator in version_evaluators
        ]
        reasonings = [
            evaluator.pred_reasonings
            for version_evaluators in self.evaluators
            for evaluator in version_evaluators
        ]
        if self.multi_system:
            table = PrettyTable()
            ans_aft = (
                [f"PREDICTED-Aft {i}" for i in range(1, self.num_dupl + 1)]
                if self.num_dupl > 1
                else ["PREDICTED-Aft"]
            )
            reas_aft = (
                [f"REASONING-Aft {i}" for i in range(1, self.num_dupl + 1)]
                if self.num_dupl > 1
                else ["REASONING-Aft"]
            )
            table.field_names = ["GOLDEN", *ans_bef, *ans_aft, *reas_bef, *reas_aft]

            attributes = zip(
                self.majority_evaluators[0].golden_answers,
                *answers,
                *reasonings,
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
            table.field_names = ["GOLDEN", *ans_bef, *reas_bef]
            attributes = zip(
                self.majority_evaluators[0].golden_answers,
                *answers,
                *reasonings,
            )

            for golden, pred_bef, reas_bef in attributes:
                table.add_row(
                    [golden, pred_bef.replace("\n", " "), wrap_text(reas_bef)]
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
        for i, features in enumerate(self.features):
            self.features[i] = sum(
                [
                    float(np.mean([result.features for result in part.results[i]]))
                    for part in self.parts
                ],
                features,
            )

    def calculate_metrics(self, **kwargs) -> None:
        """
        Calls all the individual metric calculating functions.
        :return: None
        """
        self.target_sent_dist = Metric(
            f"Target Sentence Distances", "target_sent_distances"
        )
        self.seen_context_lengths = Metric(
            f"Seen Context Lengths", "seen_context_lengths"
        )
        sample_part_lengths = []
        for part in self.parts:
            context_length = len(part.context_line_nums)
            if context_length:
                if part.context_line_nums[0] == 1:
                    sample_part_lengths = []
                sample_part_lengths.append(context_length)
            self.seen_context_lengths.add(sum(sample_part_lengths))
            # How far the target is from the question
            target_sent_dist = round(
                part.context_line_nums[-1] - mean(part.supporting_sent_inx), 2
            )
            self.target_sent_dist.add(target_sent_dist)
        # TODO: double-check
        self.sample_length = self.seen_context_lengths.all[-1]

        for version_evaluators in self.evaluators:
            for evaluator in version_evaluators:
                evaluator.calculate_accuracies()
                evaluator.calculate_attention()
                if self.run_with_reasoning:
                    evaluator.calculate_bleu()
                    evaluator.calculate_rouge()
                    evaluator.calculate_meteor()


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
        self.samples: list[Sample] = []
        self.parts: list[SamplePart] = []

        self.multi_system = multi_system
        self.versions: list[str] = (
            ["before", "after"] if self.multi_system else ["before"]
        )

        self.num_dupl: int = self.parts[0].number_of_duplicates() if self.parts else 1
        evaluator_before = [
            MetricEvaluator(level="task", version="before")
        ] * self.num_dupl
        self.evaluators: list[list[MetricEvaluator]] = (
            [
                evaluator_before,
                [MetricEvaluator(level="task", version="after")] * self.num_dupl,
            ]
            if self.multi_system
            else [evaluator_before]
        )

        features_before = Features(0, 0, 0, 0, 0, version="before")
        self.features: list[Features] = (
            [features_before, Features(0, 0, 0, 0, 0, version="after")]
            if self.multi_system
            else [features_before]
        )
        self.results: list[dict[str, str | int | float]] = []

    def add_sample(self, sample: Sample) -> None:
        """
        Add a sample to the task.

        :param sample: the sample to addition
        :return: None
        """
        self.samples.append(sample)
        for task_evaluators, sample_evaluators in zip(
            self.evaluators, sample.evaluators
        ):
            for task_evaluator, sample_evaluator in zip(
                task_evaluators, sample_evaluators
            ):
                task_evaluator.update(sample_evaluator)

    def set_results(self) -> None:
        """
        Set the results for the task by combining the results from the parts of all samples.
        Additionally, calculate the mean of the accuracy metrics and addition them to the results.

        :return: None
        """
        self.parts = [part for sample in self.samples for part in sample.parts]
        self.results: list[dict] = [part.get_result() for part in self.parts]
        for i, features in enumerate(self.features):
            self.features[i] = sum(
                [
                    float(np.mean([result.features for result in part.results[i]]))
                    for part in self.parts
                ],
                features,
            )

        for version_evaluators, version in zip(self.evaluators, self.versions):
            mean_em_accuracies, mean_sm_accuracies = [], []
            for evaluator in version_evaluators:
                mean_em_accuracies.append(evaluator.exact_match_accuracy.get_mean())
                mean_sm_accuracies.append(evaluator.soft_match_accuracy.get_mean())
            self.results[0][f"exact_match_accuracy_{version}"] = mean_em_accuracies
            self.results[0][f"soft_match_accuracy_{version}"] = mean_sm_accuracies

    def calculate_metrics(self) -> dict:
        """
        Calculate the metrics for the task.
        :return: correlation matrix of the metrics.
        """
        corr_matrices = {}
        self.seen_context_lengths = Metric(
            "Seen Context Lengths", "seen_context_lengths"
        )
        self.parts_target_distances = Metric(
            "Parts Target Sentence Distances", "parts_target_distances"
        )
        # sample_lengths and parts_answer_in_self are used on the split level
        self.sample_lengths = Metric("Sample Lengths", "sample_lengths")
        self.parts_answer_in_self = Metric(
            "Parts Answer in Self", "part_answer_in_self"
        )
        # dict.fromkeys(self.features[i].get().keys(), [])
        self.parts_features = {version: defaultdict(list) for version in self.versions}
        for sample in self.samples:
            self.seen_context_lengths.add(sample.seen_context_lengths.all)
            self.sample_lengths.add(sample.sample_length)
            self.parts_target_distances.add(sample.target_sent_dist.all)
            for part in sample.parts:
                self.parts_answer_in_self.add(part.answer_lies_in_self)
            for i, evaluator in enumerate(self.evaluators):
                sample_id_attn = {
                    (self.task_id, part.sample_id, part.part_id): part.results[
                        i
                    ].interpretability.attn_on_target
                    for part in sample.parts
                }
                evaluator.ids_with_attn_on_target.update(sample_id_attn)

        for i, (version, evaluator) in enumerate(zip(self.versions, self.evaluators)):
            for part in self.parts:
                evaluator.parts_answer_correct.add(part.results[i].answer_correct)
                evaluator.parts_max_supp_attn.add(
                    part.results[i].interpretability.max_supp_attn
                )
                evaluator.parts_attn_on_target.add(
                    part.results[i].interpretability.attn_on_target
                )
                for k, v in part.results[i].features.get().items():
                    self.parts_features[version][k].append(v)

            try:
                assert (
                    len(evaluator.parts_answer_correct)
                    == len(evaluator.parts_max_supp_attn)
                    == len(evaluator.parts_attn_on_target)
                    == len(self.seen_context_lengths)
                    == len(self.parts_answer_in_self)
                    == len(self.parts_target_distances)
                    == len(
                        self.parts_features[version][
                            list(self.parts_features[version].keys())[0]
                        ]
                    )
                )
            except AssertionError:
                print("Length mismatch in task metrics calculation:")
                print(f"parts_answer_correct: {len(evaluator.parts_answer_correct)}")
                print(f"parts_max_supp_attn: {len(evaluator.parts_max_supp_attn)}")
                print(f"parts_attn_on_target: {len(evaluator.parts_attn_on_target)}")
                print(f"seen_context_lengths: {len(self.seen_context_lengths)}")
                print(f"parts_answer_in_self: {len(self.parts_answer_in_self)}")
                print(f"parts_target_distances: {len(self.parts_target_distances)}")
                print(
                    f"parts_features: {len(self.parts_features[version][list(self.parts_features[version].keys())[0]])}"
                )

            # map the reasoning scores to the identifiers
            for score_name, flat_score_name in REASONING_SCORE_MAP.items():
                for sample in self.samples:
                    score = sample.evaluators[i].__getattribute__(score_name).all
                    ids_with_scores = {
                        (self.task_id, sample.sample_id, j + 1): value
                        for j, value in enumerate(score)
                    }
                    flat_score = evaluator.__getattribute__(flat_score_name)
                    evaluator.__setattr__(
                        flat_score_name, {**flat_score, **ids_with_scores}
                    )

            # Calculate correlation using mean part attention scores for samples
            corr_matrices[version] = evaluator.calculate_correlation(
                parts_answer_correct=evaluator.parts_answer_correct.all,
                max_supp_attn=evaluator.parts_max_supp_attn.all,
                attn_on_target=evaluator.parts_attn_on_target.all,
                seen_context_lengths=self.seen_context_lengths.all,
                target_sent_distances=self.parts_target_distances.all,
            )
        # TODO: to integrate!
        # initial = {"task_id": self.task_id}
        # for version_evaluators in self.evaluators:
        #     for evaluator in version_evaluators:
        #         evaluator.calculate_std()
        #         self.metrics.append({**initial, **evaluator.get_metrics()})
        return corr_matrices


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
        self.tasks: list[Task] = []

        self.multi_system: bool = multi_system
        self.versions: list[str] = (
            ["before", "after"] if self.multi_system else ["before"]
        )

        features_before = Features(0, 0, 0, 0, 0, version="before")
        self.features: list[Features] = (
            [features_before, Features(0, 0, 0, 0, 0, version="after")]
            if self.multi_system
            else [features_before]
        )

        self.num_dupl: int = self.tasks[0].num_dupl if self.tasks else 1
        evaluator_before = [
            MetricEvaluator(level="split", version="before")
        ] * self.num_dupl
        self.evaluators: list[list[MetricEvaluator]] = (
            [
                evaluator_before,
                [MetricEvaluator(level="split", version="after")] * self.num_dupl,
            ]
            if self.multi_system
            else [evaluator_before]
        )

    def add_task(self, task: Task) -> None:
        """
        Add a task to the prompt level data.

        :param task: the task to addition
        :return: None
        """
        self.tasks.append(task)
        for features, other_features in zip(self.features, task.features):
            features += other_features
        for evaluators, other_evaluators in zip(self.evaluators, task.evaluators):
            for evaluator, other_evaluator in zip(evaluators, other_evaluators):
                evaluator.update(other_evaluator)

    def calculate_metrics(self) -> dict:
        """
        Calculate the metrics for the split.
        :return: The correlation matrix of the metrics.
        """
        # TODO: to integrate!
        corr_matrices = {}
        for i, (version, evaluator) in enumerate(zip(self.versions, self.evaluators)):
            exact_match_accuracies = []
            max_supp_attns = []
            attn_on_targets = []
            sample_lengths = []

            self.seen_context_lengths = []
            self.parts_target_distances = []
            self.parts_answer_in_self = []

            for task in self.tasks:
                exact_match_accuracies.extend(
                    task.evaluators[i].exact_match_accuracy.all
                )
                max_supp_attns.extend(task.evaluators[i].max_supp_attn.all)
                attn_on_targets.extend(task.evaluators[i].attn_on_target.all)
                sample_lengths.extend(task.sample_lengths)

                self.seen_context_lengths.extend(task.seen_context_lengths.all)
                self.parts_target_distances.extend(task.parts_target_distances.all)
                self.parts_answer_in_self.extend(task.parts_answer_in_self.all)

                evaluator.parts_answer_correct.add(
                    task.evaluators[i].parts_answer_correct.all
                )
                evaluator.parts_attn_on_target.add(
                    task.evaluators[i].parts_attn_on_target.all
                )
                evaluator.parts_max_supp_attn.add(
                    task.evaluators[i].parts_max_supp_attn.all
                )
                evaluator.ids_with_attn_on_target.update(
                    task.evaluators[i].ids_with_attn_on_target
                )
                # self.task_features.extend(task.parts_features)

                # get the reasoning scores
                for flat_score_name in REASONING_SCORE_MAP.values():
                    task_ids_with_scores = task.evaluators[i].__getattribute__(
                        flat_score_name
                    )
                    split_ids_with_scores = evaluator.__getattribute__(flat_score_name)
                    evaluator.__setattr__(
                        flat_score_name,
                        {**split_ids_with_scores, **task_ids_with_scores},
                    )
            try:
                assert (
                    len(evaluator.parts_answer_correct)
                    == len(self.seen_context_lengths)
                    == len(self.parts_target_distances)
                    == len(self.parts_answer_in_self)
                )
            except AssertionError:
                print("Length mismatch in split metrics calculation:")
                print(f"parts_answer_correct: {len(evaluator.parts_answer_correct)}")
                print(f"seen_context_lengths: {len(self.seen_context_lengths)}")
                print(f"parts_target_distances: {len(self.parts_target_distances)}")
                print(f"parts_answer_in_self: {len(self.parts_answer_in_self)}")
            try:
                assert (
                    len(exact_match_accuracies)
                    == len(max_supp_attns)
                    == len(attn_on_targets)
                    == len(sample_lengths)
                )
            except AssertionError:
                print("Length mismatch in split metrics calculation:")
                print(f"exact_match_accuracies: {len(exact_match_accuracies)}")
                print(f"max_supp_attns: {len(max_supp_attns)}")
                print(f"attn_on_targets: {len(attn_on_targets)}")
                print(f"sample_lengths: {len(sample_lengths)}")

            corr_matrices[version] = evaluator.calculate_correlation(
                exact_match_accuracy=exact_match_accuracies,
                max_supp_attn=max_supp_attns,
                attn_on_target=attn_on_targets,
                sample_lengths=sample_lengths,
            )
        return corr_matrices


def print_metrics(data_level: Sample | Task | Split) -> None:
    """
    Print the metrics from the evaluator in a pretty table.

    :param data_level: the data level to print the metrics for
    :return: None
    """
    id_ = None
    if type(data_level) is Sample:
        id_ = data_level.sample_id
    elif type(data_level) is Task:
        id_ = data_level.task_id
    elif type(data_level) is Split:
        id_ = data_level.name

    print_metrics_table(
        evaluators=[
            evaluators[0] for evaluators in data_level.evaluators if evaluators
        ],
        id_=id_,
    )
