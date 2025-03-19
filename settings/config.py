from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional


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


@dataclass
class Student(Model):
    name: str
    max_new_tokens: int
    temperature: float
    to_continue: bool
    mode: Mode


@dataclass
class Teacher(Model):
    name: str
    max_new_tokens: int
    temperature: float
    to_continue: bool
    mode: Mode


@dataclass
class Setting:
    name: str
    interpretability: bool


@dataclass
class DataSplits:
    train: bool
    valid: bool
    test: bool


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
    id_: str = "id"
    task_id: str = "task_id"
    sample_id: str = "sample_id"
    part_id: str = "part_id"
    task: str = "task"
    golden_answer: str = "golden_answer"
    silver_reasoning: str = "silver_reasoning"
    model_answer_after: str = "model_answer_after"
    correct_after: str = "correct_after"
    model_reasoning_after: str = "model_reasoning_after"
    model_output_after: str = "model_output_after"
    exact_match_accuracy_after: str = "exact_match_accuracy_after"
    soft_match_accuracy_after: str = "soft_match_accuracy_after"
    there_after: str = "there_after"
    verbs_after: str = "verbs_after"
    pronouns_after: str = "pronouns_after"
    not_mentioned_after: str = "not_mentioned_after"
    model_answer_before: str = "model_answer_before"
    correct_before: str = "correct_before"
    model_reasoning_before: str = "model_reasoning_before"
    model_output_before: str = "model_output_before"
    exact_match_accuracy_before: str = "exact_match_accuracy_before"
    soft_match_accuracy_before: str = "soft_match_accuracy_before"
    there_before: str = "there_before"
    verbs_before: str = "verbs_before"
    pronouns_before: str = "pronouns_before"
    not_mentioned_before: str = "not_mentioned_before"


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
