# This script aggregates multiple CSV result files from different runs into a single CSV file
# by performing a majority vote on the model answers for each unique task/sample/part combination.
# It also handles numeric columns by averaging them and can optionally combine or select reasoning
# texts. The output includes metadata about the aggregation process, such as the number of
# duplicates and answer distribution.
# The script was created with Perplexity.

from __future__ import annotations

import argparse
import csv
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from data.DataLoader import DataLoader
from data.DataSaver import DataSaver
from inference.utils import majority_vote

PREFIX = Path.cwd()
while PREFIX.name != "research-project" and PREFIX.parent != PREFIX:
    PREFIX = PREFIX.parent

KEY_COLUMNS = ["task_id", "sample_id", "part_id"]
TEXT_NONE_VALUES = {None, "", "None", "nan", "NaN"}


def extract_split(path: str) -> str:
    for split in ["valid", "test", "train"]:
        if split in path:
            return split
    return "split"


def normalize_answer(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in {"None", "nan", "NaN"} else text


def normalize_reasoning(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in {"None", "nan", "NaN"} else text


def is_number(value) -> bool:
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


def choose_reasoning(reasonings: list[str], mode: str) -> str:
    cleaned = [r for r in (normalize_reasoning(r) for r in reasonings) if r]
    if not cleaned:
        return ""
    if mode == "remove":
        return ""
    if mode == "majority":
        counts = Counter(cleaned)
        max_count = max(counts.values())
        winners = [text for text, count in counts.items() if count == max_count]
        return sorted(winners)[0]
    unique = []
    seen = set()
    for r in cleaned:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return "\n---\n".join(unique)


def select_output(rows: list[dict], chosen_answer: str) -> str:
    for row in rows:
        if normalize_answer(row.get("model_answer")) == chosen_answer:
            output = row.get("model_output", "")
            return "" if output is None else str(output)
    first = rows[0].get("model_output", "")
    return "" if first is None else str(first)


def aggregate_group(rows: list[dict], reasoning_mode: str, warn_on_split: bool) -> dict:
    rows = sorted(rows, key=lambda r: str(r.get("source_run", "")))
    base = dict(rows[0])

    answers = [normalize_answer(r.get("model_answer")) for r in rows]
    nonempty_answers = [a for a in answers if a]
    counts = Counter(nonempty_answers)
    unique_answer_count = len(counts)
    answer_entropy = entropy_from_counts(counts.values())

    chosen_answer = majority_vote(nonempty_answers) if nonempty_answers else ""
    if not chosen_answer and counts:
        max_count = max(counts.values())
        winners = sorted([ans for ans, cnt in counts.items() if cnt == max_count])
        chosen_answer = winners[0]
    elif not chosen_answer and answers:
        chosen_answer = answers[0]

    split_vote = False
    if counts:
        top = counts.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            split_vote = True

    if split_vote and warn_on_split:
        key = tuple(base.get(k) for k in KEY_COLUMNS)
        warnings.warn(
            f"Split vote detected for {key}: {dict(counts)}. Chosen answer: {chosen_answer!r}",
            stacklevel=2,
        )

    numeric_columns = []
    for col in base.keys():
        if col in {"model_answer", "model_reasoning", "model_output", "source_run"}:
            continue
        numeric_values = [safe_float(r.get(col)) for r in rows]
        numeric_values = [v for v in numeric_values if v is not None]
        if numeric_values and len(numeric_values) == len(rows):
            numeric_columns.append(col)

    aggregated = dict(base)
    for col in numeric_columns:
        values = [safe_float(r.get(col)) for r in rows]
        aggregated[col] = round(sum(values) / len(values), 6)

    aggregated["model_answer"] = chosen_answer
    aggregated["model_reasoning"] = choose_reasoning(
        [r.get("model_reasoning", "") for r in rows], reasoning_mode
    )
    aggregated["model_output"] = select_output(rows, chosen_answer)
    aggregated["duplicate_count"] = len(rows)
    aggregated["num_answer_versions"] = unique_answer_count
    aggregated["answer_entropy"] = answer_entropy
    aggregated["answer_distribution"] = str(dict(counts))
    aggregated["split_vote_warning"] = split_vote
    aggregated["aggregated_from_runs"] = "|".join(
        str(r.get("source_run", "")) for r in rows
    )
    aggregated["interpretability_aggregated"] = False

    for col in list(aggregated.keys()):
        if "attn" in col.lower() or "interpret" in col.lower() or "heat" in col.lower():
            aggregated.pop(col, None)

    return aggregated


def load_csv_rows(path: str) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run(
    results_paths: list[str],
    save_path: str,
    samples_per_task: int,
    reasoning_mode: str = "remove",
    warn_on_split: bool = True,
) -> Path:
    if not results_paths:
        raise ValueError("Please provide at least one results path.")

    loader = DataLoader(prefix=PREFIX, samples_per_task=samples_per_task)
    data_split = extract_split(results_paths[0])
    saver = DataSaver(
        save_to=str(Path(save_path) / "averaged"), loaded_baseline_results=False
    )

    try:
        loader.load_results(
            results_paths=results_paths,
            data_path="../tasks_1-20_v1-2/en-valid/",
            split=data_split,
            as_parts=True,
        )
    except Exception as exc:
        warnings.warn(
            f"Project loader check failed but CSV aggregation will continue: {exc}"
        )

    all_rows = []
    for idx, path in enumerate(results_paths, start=1):
        rows = load_csv_rows(path)
        for row in rows:
            row = dict(row)
            row["source_run"] = Path(path).stem or f"run_{idx}"
            all_rows.append(row)

    grouped = defaultdict(list)
    for row in all_rows:
        key = tuple(row.get(k) for k in KEY_COLUMNS)
        grouped[key].append(row)

    aggregated_rows = [
        aggregate_group(
            rows, reasoning_mode=reasoning_mode, warn_on_split=warn_on_split
        )
        for _, rows in sorted(grouped.items())
    ]

    if not aggregated_rows:
        raise ValueError("No rows found to aggregate.")

    fieldnames = []
    seen = set()
    for row in aggregated_rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    output_name = f"{Path(results_paths[0]).stem}_average.csv"
    saver.save_output(data=aggregated_rows, headers=fieldnames, file_name=output_name)
    return saver.run_path / output_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate duplicate result CSVs into one majority-vote average CSV without interpretability."
    )
    parser.add_argument(
        "--results_paths", nargs="+", required=True, help="Input result CSV paths."
    )
    parser.add_argument("--save_path", required=True, help="Output base directory.")
    parser.add_argument("--samples_per_task", type=int, default=1)
    parser.add_argument(
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
    # python average_runs.py --results_paths run1.csv run2.csv run3.csv run4.csv --save_path /your/output/dir --samples_per_task 1 --reasoning_mode remove
    args = parse_args()
    output_path = run(
        results_paths=args.results_paths,
        save_path=args.save_path,
        samples_per_task=args.samples_per_task,
        reasoning_mode=args.reasoning_mode,
        warn_on_split=not args.no_warn_on_split,
    )
    print(f"Saved aggregated results to: {output_path}")
