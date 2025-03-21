from data.DataLoader import DataLoader
from evaluation.Evaluator import AnswerEvaluator
from inference.DataLevels import SamplePart, Results, Task, Sample


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


def run(da_path: str, reasoning_path: str, headers: dict[str, list[str]]):
    """
    Run the evaluation pipeline.

    :param da_path: Path to the data for direct answer run
    :param reasoning_path: Path to the data for reasoning run
    :return: None
    """
    loader = DataLoader()

    headers_results_before = [f"{result}_before" for result in headers["results"]]
    headers_results_after = [f"{result}_after" for result in headers["results"]]
    all_headers = (
        headers["general"]
        + headers_results_before
        + headers["results"]
        + headers_results_after
    )

    if not (da_path or reasoning_path):
        raise ValueError("Both paths should be provided.")

    da_data = loader.load_result_data(da_path, headers=all_headers, list_output=True)
    reasoning_data = loader.load_result_data(
        reasoning_path, headers=all_headers, list_output=True
    )

    if len(da_data) != len(reasoning_data):
        raise ValueError("The number of rows in the data files should be the same.")

    # TODO: load silver reasoning

    # TODO: load the interpretability results

    parts = []
    for row_da, row_reasoning in (da_data, reasoning_data):
        assert type(row_da) == dict
        assert type(row_reasoning) == dict

        # Add missing columns
        row["part_id"] = 0 if "part_id" not in row else row["part_id"]
        row["silver_reasoning"] = (
            "" if "silver_reasoning" not in row else row["silver_reasoning"]
        )
        row["model_output"] = "" if "model_output" not in row else row["model_output"]

        remove_unnecessary_columns(row)
        # Remove unnecessary columns

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
        if "model_answer_after" in row:
            part.result_after = Results(
                **{result: row[result] for result in headers_results_after}
            )
        else:
            part.result_after = Results(
                **{result: row[result] for result in headers["results"]}
            )

        parts.append(part)
        print(part.result_after)

        break

    samples = []
    tasks = []
    for idx, part in enumerate(parts):
        if part.part_id == 1:
            sample_evaluator = AnswerEvaluator(level="sample")
            sample = Sample(
                task_id=part.task_id,
                sample_id=part.sample_id,
                evaluator=sample_evaluator,
            )
            samples.append(sample)

            if part.sample_id == 1:
                task = Task(part.task_id)
                tasks.append(task)

            task.add_sample(sample)
        # if the first part or the sample id is different from the previous part or it is the first sample
        if idx == 0 or parts[idx - 1].sample_id != parts[idx].sample_id:
            sample_evaluator = AnswerEvaluator(level="sample")
            sample = Sample(
                task_id=part.task_id,
                sample_id=part.sample_id,
                evaluator=sample_evaluator,
            )
            samples.append(sample)

            if part.sample_id == 1:
                task = Task(part.task_id)
                tasks.append(task)

            task.add_sample(sample)

        sample.add_part(part)

    pass


if __name__ == "__main__":
    da_data_path = (
        "test/test_join/joined_data/valid_prompt_init_prompt_da_reasoning_results.csv"
    )
    reasoning_data_path = ""
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
    run(da_path=da_data_path, reasoning_path=reasoning_data_path, headers=headers)
