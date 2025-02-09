import os
from pathlib import Path

script_dir = Path(__file__).resolve().parent.parent
os.chdir(script_dir)

from settings.utils import expand_cardinal_points
from data.DataLoader import DataLoader


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
    lines = sorted(contexts + questions)
    answers_ = answers.copy()

    enumerated_parts = [""]
    not_enumerated_parts = [""]

    for id_, line in lines:
        if "?" in line:
            enumerated_parts[-1] += f"Question:\n{id_}. {line}\n"
            enumerated_parts[-1] += f"Answer: {answers_.pop(0)}"
            enumerated_parts.append("")

            not_enumerated_parts[-1] += f"Question:\n{line}\n"
            not_enumerated_parts[-1] += f"Answer: {answers.pop(0)}"
            not_enumerated_parts.append("")
        else:
            if not enumerated_parts[-1]:
                enumerated_parts[-1] += f"Context sentences:\n{id_}. {line}\n"
                not_enumerated_parts[-1] += f"Context sentences:\n{line}\n"
            else:
                enumerated_parts[-1] += f"{id_}. {line}\n"
                not_enumerated_parts[-1] += f"{line}\n"

    enumerated_ex = "\n\n".join(enumerated_parts)
    not_enumerated_ex = "\n\n".join(not_enumerated_parts)

    return enumerated_ex, not_enumerated_ex


def save_example(
    directory: str, examples: tuple[str, str], task: int, example_num: int
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
        Path(directory) / "enumerated" / file_name, "w", encoding="utf-8"
    ) as file:
        file.write(examples[0].strip())
    with open(
        Path(directory) / "not_enumerated" / file_name, "w", encoding="utf-8"
    ) as file:
        file.write(examples[1].strip())

    print(f"Example {example_num} for task {task} saved.")


def produce_example_files(directory: str) -> None:
    """
    Produce example files for all the tasks.

    :param directory: the directory to save the files
    """
    data_loader = DataLoader()
    data = data_loader.load_result_data(
        str(Path(directory) / "task_examples.csv"),
        ["line_id", "context/question", "golden_answer"],
        list_output=True,
    )

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


if __name__ == "__main__":
    path = "/data/examples/"
    # to reproduce the example files
    produce_example_files(directory=path)
