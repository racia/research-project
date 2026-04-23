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
from inference.utils import majority_vote
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

    averaged_row = {}
    for attr in rows[0].keys(): # All rows should have the same keys, so we can just take the keys from the first row to iterate over.
        values = [r.get(attr, "") for r in rows]
        if attr in irrelevant_columns:
            continue
        if attr in [*data_attrs, *id_columns]:
            averaged_row[attr] = rows[0][attr] # Store the first value for these columns, as they should be the same across duplicates
            continue
        if all(is_number(v) for v in values):
            numeric_values = [safe_float(v) if is_number(v) else 0 for v in values]
            averaged_row[attr] = round(mean(numeric_values), 6)
        elif "model_answer" in attr:
            answers = [clean(v) if v else "" for v in values]
            averaged_row[f"{attr}_entropy"] = entropy_from_values(answers)
            averaged_row["number_of_answer_options"] = len(set(answers))
            answer = majority_vote(answers)
            if type(answer) is list and len(answer) > 1:
                averaged_row["split_vote"] = True
                averaged_row[attr] = "\n".join(answer)
            else:
                averaged_row[attr] = answer
        else:
            warnings.warn(
                f"Unexpected non-numeric attribute in aggregation: '{attr}' with value '''{rows[0][attr]}'''"
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
        loaded = loader.load_results(
            results_path=path,
            data_path="../tasks_1-20_v1-2/en-valid/",
            split=data_split,
            as_parts=False,
            list_output=True,
        )
        if not loaded or not isinstance(loaded, tuple) or not loaded[0] or not isinstance(loaded[0], list) or not loaded[0][0] or not isinstance(loaded[0][0], dict):
            raise ValueError(f"Unexpected format in loaded results from {path}. Expected a list of lists of dicts. Got: {type(loaded)} with content: {loaded[0][0]}")
        
        duplicated_results.append(loaded[0]) # Assuming the structure is ( [ {result dict}, ... ] ) and we want the inner list of dicts for each path

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
        rows = [result[id_] for result in duplicated_results]
        aggregated = aggregate(rows)
        saver.save_output(
            data=[aggregated],
            headers=aggregated.keys(),
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
    parser.add_argument( # This argument is currently not used in the code, but it can be implemented in the future to handle reasoning texts according to the specified mode.
        "--reasoning_mode",
        choices=["remove", "join", "majority"],
        default="remove",
        help="How to handle duplicate reasonings.",
    )
    parser.add_argument(
        "--no_warn_on_split",
        action="store_true",
        help="Disable warnings for tied majority votes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # This script doesn't calculate metrics!
    # It should therefore be run after the initial evaluation of the duplicating runs to record them correctly.
    # python average_runs.py --results_paths run1.csv run2.csv run3.csv run4.csv --save_path /your/output/dir --samples_per_task 1
    # args = parse_args()
    results_paths = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v1/all_tasks_joined/joined_direct_answer_results.csv",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v2/all_tasks_joined/joined_direct_answer_results.csv",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v3/all_tasks_joined/joined_direct_answer_results.csv",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v4/all_tasks_joined/joined_direct_answer_results.csv",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v5/all_tasks_joined/joined_direct_answer_results.csv",
    ]
    save_path = "outputs/test-average/" #"/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/test-average/"
    samples_per_task = 3
    output_path = run(
        results_paths=results_paths,
        save_path=save_path,
        samples_per_task=samples_per_task,
    )
    print(f"Saved aggregated results to: {output_path}")
