from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass
class Model:
    name: str
    mode: str
    max_new_tokens: int
    temperature: float
    to_continue: bool


@dataclass
class DataSplits:
    train: str = "train"
    valid: str = "valid"
    test: str = "test"


@dataclass
class Enumerate:
    context: str = "context"
    question: str = "question"


@dataclass
class Data:
    path: str
    splits: dict[DataSplits, bool]
    task_ids: bool | list[int]
    samples_per_task: int
    to_enumerate: dict[Enumerate, bool]


@dataclass
class Wrapper:
    context: str = "context"
    question: str = "question"
    answer: str = "answer"


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
    wrapper: dict[Wrapper, str]
    examples: dict[Examples, Union[bool, int, str]]


@dataclass
class CSVHeaders:
    id_: str = "id"
    task_id: str = "task_id"
    sample_no: str = "sample_no"
    part: str = "part"
    true_answer: str = "true_answer"
    model_answer: str = "model_answer"
    correct: str = "correct"
    model_reasoning: str = "model_reasoning"
    model_output: str = "model_output"
    exact_match_accuracy: str = "exact_match_accuracy"
    soft_match_accuracy: str = "soft_match_accuracy"
    there: str = "there"
    verbs: str = "verbs"
    pronouns: str = "pronouns"
    not_mentioned: str = "not_mentioned"
    attn_scores: str = "attn_scores"
    x_tokens: str = "x_tokens"
    y_tokens: str = "y_tokens"

@dataclass
class Logging:
    print_to_file: bool


@dataclass
class Results:
    headers: list[CSVHeaders]
    save_to: str


@dataclass
class Config:
    model: Model
    data: Data
    prompt: Prompt
    logging: Logging
    results: Results

@dataclass
class Interpretability:
    model: Model
    path: str
