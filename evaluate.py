from pathlib import Path

from data.DataLoader import DataLoader
from data.DataSaver import DataSaver
from evaluation.Evaluator import AnswerEvaluator, MetricEvaluator
from inference.DataLevels import SamplePart, Results, Task, Sample, Split


def mean_attn_score(attn_scores: list[float]) -> float:
    """
    Calculate the mean attention score.

    :param attn_scores: The attention scores.
    :return: The mean attention score.
    """
    return sum(attn_scores) / len(attn_scores)


def remove_unnecessary_columns(row: dict) -> None:
    """
    Remove unnecessary columns from the row.

    :param row: The row to remove the columns from.
    :return: None
    """
    unnecessary_columns = [
        "correct",
        "correct?",
        "exact_match_accuracy",
        "soft_match_accuracy",
        "there",
        "verbs",
        "pronouns",
        "not_mentioned",
    ]
    for col in unnecessary_columns:
        if col in row:
            del row[col]


def add_part_ids(parts: list[SamplePart]) -> list[SamplePart]:
    """
    Add part ids to the parts.

    :param parts: The parts to add the path ids to.
    :return: The parts with the path ids.
    """
    part_id = 1
    for inx, part in enumerate(parts):
        if parts[inx - 1].sample_id != part.sample_id:
            part_id = 1
        if part.part_id == 0:
            part.part_id = part_id
            part_id += 1
    return parts


def extract_split(data_path) -> str:
    """
    Extract the split from the data path. If the split is not found, return "split".

    :param data_path: The path to the data.
    :return: The split.
    """
    for split in ["valid", "test", "train"]:
        if split in data_path:
            return split
    return "split"


def run(data_path: str, headers: dict[str, list[str]], save_path: str) -> None:
    """
    Run the evaluation pipeline.

    :param data_path: Path to the data for evaluation
    :param headers: The headers for the data
    :param save_path: Path to save the results
    :return: None
    """
    loader = DataLoader()
    saver = DataSaver(save_to=save_path)

    data_split = extract_split(data_path)

    headers_results_before = [f"{result}_before" for result in headers["results"]]
    headers_results_after = [f"{result}_after" for result in headers["results"]]
    all_headers = (
        headers["general"]
        + headers_results_before
        + headers["results"]
        + headers_results_after
    )

    data = loader.load_result_data(data_path, headers=all_headers, list_output=True)

    # TODO: Make sure that the reasoning data corresponds to the results data
    silver_reasoning_path = "data/golden_reasoning_1.csv"
    silver_reasoning_headers = [
        "task_id",
        "sample_id",
        "part_id",
        "context",
        "question",
        "answer",
        "reasoning",
    ]
    silver_reasoning_data = loader.load_result_data(
        silver_reasoning_path,
        headers=silver_reasoning_headers,
        list_output=True,
        sep=",",
    )

    # TODO: load the interpretability results

    before = None
    general = None

    parts = []
    for row, row_sil in zip(data, silver_reasoning_data):
        assert type(row) == dict

        # Add missing columns
        row["part_id"] = 0 if "part_id" not in row else row["part_id"]
        row["model_output"] = "" if "model_output" not in row else row["model_output"]

        if row["golden_answer"] == row_sil["answer"]:
            row["silver_reasoning"] = row_sil["reasoning"]
        else:
            raise ValueError(
                "The golden answer in the results does not match the answer in the silver reasoning data."
            )

        remove_unnecessary_columns(row)

        print("Row:")
        print(row)

        for gen_header in headers["general"]:
            if gen_header in row:
                print(f"{gen_header}: {row[gen_header]}")

        part = SamplePart(
            **{gen_header: row[gen_header] for gen_header in headers["general"]}
        )
        if "model_answer_before" in row:
            part.result_before = Results(
                **{result: row[result] for result in headers_results_before}
            )
            before = True
        if "model_answer_after" in row:
            part.result_after = Results(
                **{result: row[result] for result in headers_results_after}
            )
        else:
            part.result_after = Results(
                **{result: row[result] for result in headers["results"]}
            )
            general = True

        parts.append(part)
        print(part.result_after)

        break

    if parts[0].part_id == 0:
        parts = add_part_ids(parts)

        path = Path(data_path)
        file_name = path.parent / f"{path.stem}_with_parts.csv"

        part_dicts = [part.get_result() for part in parts]
        print(part_dicts[0])
        print(all_headers)

        saver.save_output(part_dicts, all_headers, file_name)

    samples = []
    tasks = []
    if before and not general:
        split_evaluator_before = MetricEvaluator(level="split")
        # TODO make a function for each repetition
        pass

    sample = False
    task = False
    split = Split(name=data_split)

    for idx, part in enumerate(parts):
        if part.sample_id == 1:
            if task:
                task.evaluator_after.print_accuracies(id_=part.task_id)
                task.set_results()

            task = Task(part.task_id)
            tasks.append(task)
            split.add_task(task)

        if part.part_id == 1:
            if sample:
                sample.print_sample_predictions()
                exact_match_acc, soft_match_acc = (
                    sample.evaluator_after.calculate_accuracies()
                )
                sample.evaluator_after.print_accuracies(
                    id_=part.sample_id,
                    exact_match_acc=exact_match_acc,
                    soft_match_acc=soft_match_acc,
                )

            sample_evaluator = AnswerEvaluator(level="sample")
            sample = Sample(
                task_id=part.task_id,
                sample_id=part.sample_id,
                evaluator=sample_evaluator,
            )

            samples.append(sample)
            task.add_sample(sample)

        sample.evaluator_after.golden_answers.append(part.golden_answer)
        sample.evaluator_after.silver_reasonings.append(part.silver_reasoning)
        sample.add_part(part)

        break

    pass


if __name__ == "__main__":
    data_path = (
        "test/test_join/joined_data/valid_prompt_init_prompt_da_reasoning_results.csv"
    )
    headers = {
        "general": [
            "id_",
            "task_id",
            "sample_id",
            "part_id",
            "task",
            "golden_answer",
            "silver_reasoning",
        ],
        "results": [
            "model_answer",
            "model_reasoning",
            "model_output",
        ],
    }
    run(data_path=data_path, headers=headers, save_path="test/test_join/joined_data/")
