from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class Mode:
    train: str = "train"
    eval: str = "eval"


@dataclass
class Model:
    name: str
    max_new_tokens: int
    temperature: float
    to_continue: bool
    mode: Mode
    interpretability: bool


@dataclass
class Student(Model):
    name: str
    max_new_tokens: int
    temperature: float
    to_continue: bool
    mode: Mode
    interpretability: bool


@dataclass
class Teacher(Model):
    name: str
    max_new_tokens: int
    temperature: float
    to_continue: bool
    mode: Mode
    p: float
    k: int
    interpretability: bool


@dataclass
class Setting:
    name: str


@dataclass
class DataSplits:
    train: bool = "train"
    valid: bool = "valid"
    test: bool = "test"

    def get(self) -> str:
        split = [split for split, value in self.__dict__.items() if value]
        if not split:
            raise ValueError(
                "No split found in the given path. Please make sure the path contains 'train', 'valid' or 'test'."
            )
        if len(split) > 1:
            warnings.warn("Multiple splits found. Using the first one.")
        return split[0]


@dataclass
class Enumerate:
    context: bool
    question: bool


@dataclass
class Wrapper:
    context: str
    question: str
    reasoning: str
    answer: str

    def __repr__(self):
        return (
            f"Wrapper(context={self.context}, question={self.question}, "
            f"reasoning={self.reasoning}, answer={self.answer})"
        )


@dataclass
class Data:
    path: str
    baseline_results: str
    splits: DataSplits
    task_ids: bool | list[int]
    samples_per_task: int
    to_enumerate: Enumerate
    wrapper: Wrapper


@dataclass
class Examples:
    add: bool
    enumerated: bool
    handpicked: bool
    not_mentioned: bool
    number: int
    wrapper: str


@dataclass
class Prompt:
    paths: list[str]


class InitPrompt:
    examples: dict[Examples, Union[bool, int, str]]


class FurtherPrompt(Prompt):
    wrapper: str


@dataclass
class CSVHeaders:
    id_: str = "id_"
    task_id: str = "task_id"
    sample_id: str = "sample_id"
    part_id: str = "part_id"
    task: str = "task"
    answer_lies_in_self: str = "answer_lies_in_self"
    golden_answer: str = "golden_answer"
    silver_reasoning: str = "silver_reasoning"
    model_answer_after: str = "model_answer_after"
    answer_correct_after: str = "answer_correct_after"
    model_reasoning_after: str = "model_reasoning_after"
    reasoning_correct_after: str = "reasoning_correct_after"
    model_output_after: str = "model_output_after"
    exact_match_accuracy_after: str = "exact_match_accuracy_after"
    soft_match_accuracy_after: str = "soft_match_accuracy_after"
    there_after: str = "there_after"
    verbs_after: str = "verbs_after"
    pronouns_after: str = "pronouns_after"
    not_mentioned_after: str = "not_mentioned_after"
    context_sents_hall_after: str = "context_sents_hall_after"
    iterations: str = "iterations"
    model_answer_before: str = "model_answer_before"
    answer_correct_before: str = "answer_correct_before"
    model_reasoning_before: str = "model_reasoning_before"
    reasoning_correct_before: str = "reasoning_correct_before"
    model_output_before: str = "model_output_before"
    exact_match_accuracy_before: str = "exact_match_accuracy_before"
    soft_match_accuracy_before: str = "soft_match_accuracy_before"
    max_supp_attn_before: str = "max_supp_attn_before"
    max_supp_attn_after: str = "max_supp_attn_after"
    attn_on_target_before: str = "attn_on_target_before"
    attn_on_target_after: str = "attn_on_target_after"
    there_before: str = "there_before"
    verbs_before: str = "verbs_before"
    pronouns_before: str = "pronouns_before"
    not_mentioned_before: str = "not_mentioned_before"
    context_sents_hall_before: str = "context_sents_hall_before"


@dataclass
class Logging:
    print_to_file: bool


@dataclass
class Results:
    headers: CSVHeaders


@dataclass
class Config:
    model: Optional[Model]
    student: Optional[Student]
    teacher: Optional[Teacher]
    setting: Setting
    data: Data
    init_prompt: Prompt
    eval_prompt: Optional[FurtherPrompt]
    resume_prompt: Optional[FurtherPrompt]
    logging: Logging
    results: Results
