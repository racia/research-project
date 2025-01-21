import os
import sys
from abc import ABC
from pathlib import Path

from baseline.utils import expand_cardinal_points

sys.path.insert(0, str(Path(Path.cwd()).parents[0]))
from data.DataLoader import DataLoader

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)


def construct_example(
    contexts: list[tuple[int, str]],
    questions: list[tuple[int, str]],
    answers: list[str],
) -> tuple[str, str]:
    """
    Construct an example from the given contexts, questions, and answers.

    :param contexts: list of tuples with the line number and the context sentence
    :param questions: list of tuples with the line number and the question
    :param answers: list of answers

    :return: the constructed example
    """
    parts = [
        (
            f"Context sentences:\n{context[0]}. {context[1]}\n"
            f"Question:\n{question[0]}. {question[1]}\n"
            f"Answer: {answer}"
        )
        for context, question, answer in zip(contexts, questions, answers)
    ]
    enumerated_ex = "\n\n".join(parts)
    parts = [
        (
            f"Context sentences:\n{context[1]}\n"
            f"Question:\n{question[1]}\n"
            f"Answer: {answer}"
        )
        for context, question, answer in zip(contexts, questions, answers)
    ]
    not_enumerated_ex = "\n\n".join(parts)
    return enumerated_ex, not_enumerated_ex


def save_example(
    directory: Path, examples: tuple[str, str], task: int, example_num: int
) -> None:
    """
    Save the example to a file.

    :param directory: the directory to save the file
    :param examples: the examples to save: enumerated and not enumerated
    :param task: the task number
    :param example_num: the example number
    """
    file_name = f"task_{task}_example_{example_num}.txt"
    with open(
        Path(directory / "enumerated" / file_name), "w", encoding="utf-8"
    ) as file:
        file.write(examples[0].strip())
    with open(
        (directory / "not_enumerated" / file_name), "w", encoding="utf-8"
    ) as file:
        file.write(examples[0].strip())

    print(f"Example {example_num} for task {task} saved.")


def produce_example_files(directory: str) -> None:
    """
    Produce example files for all the tasks.

    :param directory: the directory to save the files
    """
    data_loader = DataLoader()
    data = data_loader.load_result_data(
        f"{directory}task_examples.csv",
        ["line_id", "context/question", "golden_answer"],
        list_output=True,
    )
    directory = Path(directory)

    for task, rows in data.items():
        example_num = 1
        contexts, questions, answers = [], [], []
        for row in rows:
            if row["line_id"] == 1:
                if questions:
                    answers = [
                        ", ".join(expand_cardinal_points(answer.split()))
                        for answer in answers
                    ]
                    enumerated_ex, not_enumerated_ex = construct_example(
                        contexts, questions, answers
                    )
                    save_example(
                        directory, (enumerated_ex, not_enumerated_ex), task, example_num
                    )
                    example_num += 1
                contexts, questions, answers = [], [], []

            if "?" not in row["context/question"]:
                contexts.append((row["line_id"], row["context/question"]))
            else:
                questions.append((row["line_id"], row["context/question"]))
                answers.append(row["golden_answer"])

        # example_num += 1
        if questions:
            answers = [
                ", ".join(expand_cardinal_points(answer.split())) for answer in answers
            ]
            enumerated_ex, not_enumerated_ex = construct_example(
                contexts, questions, answers
            )
            save_example(
                directory, (enumerated_ex, not_enumerated_ex), task, example_num
            )


class Task(ABC):
    """
    A class to represent a task.
    """

    def __init__(self, number: int, to_enumerate: bool = True):
        self.number = number
        self.to_enumerate = to_enumerate
        self.wrapper = "*EXAMPLE{number}*\n___________\n{example}\n___________\n"
        self.folder = "enumerated/" if to_enumerate else "not_enumerated/"

    def __repr__(self):
        raise NotImplementedError(
            "The __repr__ method is not implemented, use TaskExample class."
        )

    def __iter__(self):
        raise NotImplementedError(
            "The __iter__ method is not implemented, use TaskExamples class."
        )


class TaskExample(Task):
    """
    A class to represent examples for a task.
    """

    def __repr__(self):
        """
        Return the first example for the task without numeration.
        """
        path = Path.cwd() / self.folder / f"task_{self.number}_example_1.txt"
        with open(path, "r", encoding="utf-8") as file:
            example = file.read()
            wrapped_example = self.wrapper.format(number="", example=example)

        return wrapped_example


class TaskExamples(Task):
    """
    A class to represent examples for a task.
    """

    def __iter__(self):
        """
        Return an iterator over all the examples for the task.
        """
        all_files = [
            file
            for file in os.listdir(self.folder)
            if file.startswith(f"task_{self.number}_")
        ]
        all_examples = []
        for i, file in enumerate(all_files, 1):
            path = Path.cwd() / self.folder / f"task_{self.number}_example_{i}.txt"
            with open(path, "r", encoding="utf-8") as file:
                example = file.read()
                wrapped_example = self.wrapper.format(number=f" {i}", example=example)
                all_examples.append(wrapped_example)
        return iter(all_examples)


if __name__ == "__main__":
    # path = "/Users/bohdana.ivakhnenko/PycharmProjects/research-project/data/examples/"
    # to reproduce the example files
    # produce_example_files(directory=path)

    # to print the first example for the task
    print(TaskExample(number=19))

    # to print all the examples for the task
    for example in TaskExamples(number=19):
        print(example)
