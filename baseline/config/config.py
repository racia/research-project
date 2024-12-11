from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Model:
    name: str
    max_new_tokens: int
    temperature: float


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
    splits: Dict[DataSplits, bool]
    task_ids: bool | List[int]
    samples_per_task: int
    to_enumerate: Dict[Enumerate, bool]


@dataclass
class Prompt:
    name: str
    text: str


@dataclass
class CSVHeaders:
    id_: str = "id"
    task_id: str = "task_id"
    sample_no: str = "sample_no"
    task: str = "task"
    true_result: str = "true_result"
    model_result: str = "model_result"
    accuracy: str = "accuracy"
    soft_accuracy: str = "soft_accuracy"


@dataclass
class Repository:
    path: str


@dataclass
class Results:
    print_to_file: bool
    path: str
    headers: List[CSVHeaders]


@dataclass
class Config:
    model: Model
    data: Data
    prompt: Prompt
    repository: Repository
    results: Results
