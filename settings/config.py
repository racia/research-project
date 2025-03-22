from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class Model:
    name: str
    max_new_tokens: int
    temperature: float
    to_continue: bool


@dataclass
class Student(Model):
    name: str
    max_new_tokens: int
    temperature: float
    to_continue: bool


@dataclass
class Teacher(Model):
    name: str
    max_new_tokens: int
    temperature: float
    to_continue: bool


@dataclass
class Setting:
    name: str


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
    model_answer: str = "model_answer"
    correct: str = "correct"
    golden_answer: str = "golden_answer"
    model_reasoning: str = "model_reasoning"
    silver_reasoning: str = "silver_reasoning"
    model_output: str = "model_output"
    exact_match_accuracy: str = "exact_match_accuracy"
    soft_match_accuracy: str = "soft_match_accuracy"
    there: str = "there"
    verbs: str = "verbs"
    pronouns: str = "pronouns"
    not_mentioned: str = "not_mentioned"


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
