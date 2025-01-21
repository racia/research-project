from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Model:
    name: str
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


@dataclass
class Prompt:
    paths: list[str]
    wrapper: dict[Wrapper, str]


@dataclass
class CSVHeaders:
    id_: str = "id"
    task_id: str = "task_id"
    sample_no: str = "sample_no"
    task: str = "task"
    true_result: str = "true_result"
    model_result: str = "model_result"
    accuracy: str = "accuracy"
    soft_match_accuracy: str = "soft_match_accuracy"


@dataclass
class Repository:
    path: str
    save_here: bool


@dataclass
class Results:
    parse: bool
    print_to_file: bool
    path: str
    headers: list[CSVHeaders]


@dataclass
class Config:
    model: Model
    data: Data
    prompt: Prompt
    repository: Repository
    results: Results
