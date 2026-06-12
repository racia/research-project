# This script aggregates multiple CSV result files from different runs into a single CSV file
# by performing a majority vote on the model answers for each unique task/sample/part combination.
# It also handles numeric columns by averaging them and can optionally combine or select reasoning
# texts. The output includes metadata about the aggregation process, such as the number of
# duplicates and answer distribution.
# The script was created with Perplexity.

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from statistics import mean
from typing import Iterable

from data.DataLoader import DataLoader
from data.DataSaver import DataSaver
from evaluation.utils import extract_split
from inference.utils import majority_vote, flatten
from settings.utils import clean

PREFIX = Path.cwd()
while PREFIX.name != "research-project" and PREFIX.parent != PREFIX:
    PREFIX = PREFIX.parent


def is_number(value) -> bool:
    """Check if the value can be converted to a float."""
    if isinstance(value, bool):
        return True
    try:
        f = float(value)
        return not math.isnan(f)
    except (TypeError, ValueError):
        return False


def safe_float(value):
    if is_number(value):
        return float(value)
    return None


def entropy_from_counts(counts: Iterable[int]) -> float:
    counts = [c for c in counts if c > 0]
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        p = c / total
        ent -= p * math.log2(p)
    return round(ent, 6)


def entropy_from_values(values: Iterable) -> float:
    from collections import Counter

    value_counts = Counter(values)
    counts = value_counts.values()
    return entropy_from_counts(counts)


def aggregate(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: str(r.get("source_run", "")))

    id_columns = ["task_id", "sample_id", "part_id", "id"]
    irrelevant_columns = [
        "reasoning_correct_before",
        "reasoning_correct_after",
        "answer_correct_before",
        "answer_correct_after",
        "exact_match_accuracy_before",
        "exact_match_accuracy_after",
        "soft_match_accuracy_before",
        "soft_match_accuracy_after",
        "silver_reasoning",
    ]
    data_attrs = ["golden_answer", "task", "answer_lies_in_self"]

    extra_keys = flatten([set(r.keys()) - set(rows[0].keys()) for r in rows])
    if any(len(v) for v in extra_keys):
        extra_keys_print = "\n- ".join([str(v) for v in extra_keys if len(v) > 0])
        warnings.warn(
            "All rows must have the same keys for aggregation. Found extra keys in some rows:\n"
            f"{extra_keys_print}"
        )
    missing_keys = flatten([set(rows[0].keys()) - set(r.keys()) for r in rows])
    if any(len(v) for v in missing_keys):
        missing_keys_print = "\n- ".join([str(v) for v in missing_keys if len(v) > 0])
        warnings.warn(
            "All rows must have the same keys for aggregation. Found missing keys in some rows:\n"
            f"{missing_keys_print}"
        )

    averaged_row = {}
    for attr in rows[0].keys():
        values = [r.get(attr, "") for r in rows]
        if attr in irrelevant_columns:
            continue
        if attr in extra_keys + missing_keys:
            continue
        if attr in [*data_attrs, *id_columns]:
            # Store the first value for these columns, as they should be the same across duplicates
            averaged_row[attr] = rows[0][attr]
            continue
        if all(is_number(v) for v in values):
            numeric_values = [safe_float(v) if is_number(v) else 0 for v in values]
            averaged_row[attr] = round(mean(numeric_values), 6)
        elif "model_answer" in attr:
            answers = [clean(v) if v else "" for v in values]
            averaged_row[f"{attr}_entropy"] = entropy_from_values(answers)
            averaged_row["number_of_answer_options"] = len(set(answers))
            answer = majority_vote(answers)
            print(f"Answer {answer!r} is chosen out of {answers!r}")
            if type(answer) is list and len(answer) > 1:
                averaged_row["split_vote"] = True
                averaged_row[attr] = "\n".join(answer)
            else:
                averaged_row["split_vote"] = False
                averaged_row[attr] = answer[0] if answer else ""
        elif "model_output" in attr or "model_reasoning" in attr:
            averaged_row[attr] = "\n\n".join([v for v in values if v])
        else:
            warnings.warn(
                f"Unexpected non-numeric attribute in aggregation: '{attr}' with value:\n'{rows[0][attr]}'"
            )

    return averaged_row


def run(
    results_paths: list[str],
    save_path: str,
    samples_per_task: int,
) -> Path:
    if not results_paths:
        raise ValueError("Please provide at least one results path.")

    loader = DataLoader(prefix=PREFIX, samples_per_task=samples_per_task)
    data_split = extract_split(results_paths[0])
    saver = DataSaver(save_to=save_path, loaded_baseline_results=False)
    ave_file_name = Path(results_paths[0]).stem + "_averaged.csv"
    output_path = saver.results_path / ave_file_name
    # raise error if the output file exists and is not empty, to avoid overwriting existing results
    if output_path.exists() and output_path.stat().st_size > 0:
        raise FileExistsError(
            f"Output file already exists and is not empty: {output_path}. Please choose a different save path or remove the existing file."
        )

    duplicated_results = []
    for path in results_paths:
        data, _ = loader.load_results(
            results_path=path,
            data_path="../tasks_1-20_v1-2/en-valid/",
            split=data_split,
            as_parts=False,
            list_output=True,
        )
        if not data:
            raise ValueError(
                f"No data loaded from {path}. Please check the file and path."
            )
        if not isinstance(data, list) or not isinstance(data[0], dict):
            raise ValueError(
                f"Unexpected format in loaded results from {path}. "
                f"Expected a list[dict] per entry. "
                f"Got: {type(data)}[{type(data[0])}] of content: {data[0]}"
            )

        duplicated_results.append(data)

    all_ids = set()
    for i, result in enumerate(duplicated_results):
        assert (
            type(result) is list
        ), f"Expected list of dicts for results, got {type(result)}\nResult content: {result}"
        mapped_results = {}
        for r in result:
            assert type(r) is dict, f"Expected dict, got {type(r)}\nResult content: {r}"
            id_ = (r.get("task_id"), r.get("sample_id"), r.get("part_id"))
            all_ids.add(id_)
            mapped_results[id_] = r
        duplicated_results[i] = mapped_results

    for id_ in sorted(list(all_ids)):
        rows = []
        for i, result in enumerate(duplicated_results):
            if id_ not in result:
                raise ValueError(
                    f"Result with id {id_} not found in path {results_paths[i]}, "
                    f"possibly missing these ids in results"
                )
            rows.append(result[id_])
        aggregated = aggregate(rows)
        saver.save_output(
            data=[aggregated],
            headers=list(aggregated.keys()),
            file_name=output_path,
        )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate duplicate result CSVs into one majority-vote average CSV without interpretability."
    )
    parser.add_argument(
        "--results_paths", nargs="+", required=True, help="Input result CSV paths."
    )
    parser.add_argument("--save_path", required=True, help="Output base directory.")
    parser.add_argument("--samples_per_task", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    # This script doesn't calculate metrics!
    # It should therefore be run after the initial evaluation of the duplicating runs to record them correctly.
    # python3 average_runs.py --results_paths run1.csv run2.csv run3.csv run4.csv --save_path /your/output/dir --samples_per_task 1
    # args = parse_args()
    # output_path = run(
    #     results_paths=args.results_paths,
    #     save_path=args.save_path,
    #     samples_per_task=args.samples_per_task,
    # )
    # TODO: Uncomment for testing
    # results_paths = [
    #     "/workspace/students/reasoning/results/basic-baseline/test/da/v1/all_tasks_joined/joined_direct_answer_results.csv",
    #     "/workspace/students/reasoning/results/basic-baseline/test/da/v2/all_tasks_joined/joined_direct_answer_results.csv",
    #     "/workspace/students/reasoning/results/basic-baseline/test/da/v3/all_tasks_joined/joined_direct_answer_results.csv",
    #     "/workspace/students/reasoning/results/basic-baseline/test/da/v4/all_tasks_joined/joined_direct_answer_results.csv",
    #     "/workspace/students/reasoning/results/basic-baseline/test/da/v5/all_tasks_joined/joined_direct_answer_results.csv",
    # ]
    # save_path = (
    #     "/workspace/students/reasoning/results/basic-baseline/test/da/average_run"
    # )
    # results_paths = [
    #     "/workspace/students/reasoning/results/basic-baseline/test/reasoning/v1/all_tasks_joined/joined_reasoning_results.csv",
    #     "/workspace/students/reasoning/results/basic-baseline/test/reasoning/v2/all_tasks_joined/joined_reasoning_results.csv",
    #     "/workspace/students/reasoning/results/basic-baseline/test/reasoning/v3/all_tasks_joined/joined_reasoning_results.csv",
    #     "/workspace/students/reasoning/results/basic-baseline/test/reasoning/v4/all_tasks_joined/joined_reasoning_results.csv",
    #     "/workspace/students/reasoning/results/basic-baseline/test/reasoning/v5/all_tasks_joined/joined_reasoning_results.csv",
    # ]
    # save_path = (
    #     "/workspace/students/reasoning/results/basic-baseline/test/reasoning/average_run"
    # )
    # results_paths = [
    #     "/workspace/students/reasoning/results/baseline/test/da/v1/all_tasks_joined/joined_direct_answer_results.csv",
    #     "/workspace/students/reasoning/results/baseline/test/da/v2/all_tasks_joined/joined_direct_answer_results.csv",
    #     "/workspace/students/reasoning/results/baseline/test/da/v3/all_tasks_joined/joined_direct_answer_results.csv",
    #     "/workspace/students/reasoning/results/baseline/test/da/v4/all_tasks_joined/joined_direct_answer_results.csv",
    #     "/workspace/students/reasoning/results/baseline/test/da/v5/all_tasks_joined/joined_direct_answer_results.csv",
    # ]
    # save_path = "/workspace/students/reasoning/results/baseline/test/da/average_run"
    # results_paths = [
    #     "/workspace/students/reasoning/results/baseline/test/reasoning/v1/all_tasks_joined/joined_reasoning_results.csv",
    #     "/workspace/students/reasoning/results/baseline/test/reasoning/v2/all_tasks_joined/joined_reasoning_results.csv",
    #     "/workspace/students/reasoning/results/baseline/test/reasoning/v3/all_tasks_joined/joined_reasoning_results.csv",
    #     "/workspace/students/reasoning/results/baseline/test/reasoning/v4/all_tasks_joined/joined_reasoning_results.csv",
    #     "/workspace/students/reasoning/results/baseline/test/reasoning/v5/all_tasks_joined/joined_reasoning_results.csv",
    # ]
    # save_path = "/workspace/students/reasoning/results/baseline/test/reasoning/average_run"
    results_paths = [
        "/workspace/students/reasoning/results/skyline/test/da/v1/all_tasks_joined/joined_direct_answer_results.csv",
        "/workspace/students/reasoning/results/skyline/test/da/v2/all_tasks_joined/joined_direct_answer_results.csv",
        "/workspace/students/reasoning/results/skyline/test/da/v3/all_tasks_joined/joined_direct_answer_results.csv",
        "/workspace/students/reasoning/results/skyline/test/da/v4/all_tasks_joined/joined_direct_answer_results.csv",
        "/workspace/students/reasoning/results/skyline/test/da/v5/all_tasks_joined/joined_direct_answer_results.csv",
    ]
    save_path = "/workspace/students/reasoning/results/skyline/test/da/average_run"
    samples_per_task = 100
    output_path = run(
        results_paths=results_paths,
        save_path=save_path,
        samples_per_task=samples_per_task,
    )
    print(f"Saved aggregated results to: {output_path}")
