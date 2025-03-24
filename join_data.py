# This script is used to join the data from the different sources into a single file.
# The choice of the source file for each task is based on the number of parts:
# the file with the most parts for a task is chosen as the source for it.
# The headers are checked for matching and the unique headers are joined.
# The data is then joined and saved together in the specified directory.
# Log files are copied to there, too, but not the metrics and plots,
# because that data depends on the run being completed.

from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path
from typing import Generator

from prettytable import PrettyTable
from spacy.tokens.doc import defaultdict

from data.DataLoader import DataLoader
from data.DataSaver import DataSaver
from plots.utils import get_paths, find_difference_in_paths

PREFIX = Path.cwd()
while PREFIX.name != "research-project":
    PREFIX = PREFIX.parent


def flatten(items: list) -> Generator:
    """
    Flatten a list of items.

    :param items: list of items
    :return: flattened list
    """
    for item in items:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def print_counts_table(id_counts: dict[int, list[int]], paths: list[Path]):
    """
    Print a table of counts of parts in tasks per path.
    """
    table = PrettyTable()

    # Set up the table columns: sample parts and task IDs
    table.field_names = ["Sample Parts"] + list(id_counts.keys())
    path_differences = find_difference_in_paths(paths)

    # Add rows for each path
    for i, name in enumerate(path_differences):
        row = [name] + [part_count[i] for part_count in id_counts.values()]
        table.add_row(row)

    print("\nCounts of samples parts for tasks per path:")
    print(table, end="\n\n")


def count_parts_per_task(data: dict[Path, dict[str, list]]) -> dict[int, list[int]]:
    """
    Count the number of parts for each task in the data.

    :param data: data from the different result files
    :return: dictionary of task IDs and counts of parts
    """
    id_counts = defaultdict(list[int])
    for path, results in data.items():
        task_ids = results["task_id"]
        for task_id in range(1, 21):
            id_counts[task_id].append(task_ids.count(task_id))
    return id_counts


def define_sources_for_tasks(
    id_counts: dict[int, list[int]], paths: list[Path]
) -> dict[Path, list[int]]:
    """
    Define the source file for each task depending on the number of parts.
    The file with the most parts for a task is chosen as the source for it.

    :param id_counts: dictionary of task IDs and counts of parts
    :param paths: list of paths to the result files
    :return: dictionary of paths and task IDs for each path
    """
    tasks = defaultdict(list)
    for task_id, counts in id_counts.items():
        max_counts = max(counts)

        if max_counts == 0:
            warnings.warn(f"No parts found for task {task_id}.")

        inx = counts.index(max_counts)
        tasks[paths[inx]].append(task_id)
    return tasks


def join_headers(all_headers: list[list[str]]) -> tuple:
    """
    Get all unique headers from the data in one tuple

    :param all_headers: list of headers from the data files
    :return: list of unique headers
    """
    headers = []
    lengths = [len(headers) for headers in all_headers]

    for i in range(max(lengths)):
        for header_list in all_headers:
            if len(header_list) >= i and header_list[i] not in headers:
                headers.append(header_list[i])
    return tuple(headers)


def get_headers(data: dict[Path, dict[str, list]]) -> tuple:
    """
    Check if the headers match in the data files and get all unique headers.

    :param data: data from the different result files
    :return: tuple of unique headers
    """
    all_headers = [[header for header in results.keys()] for results in data.values()]

    set_headers = [set(headers) for headers in all_headers]
    set_lengths = [len(headers) for headers in set_headers]
    fewest_headers_no = min(set_lengths)
    fewest_headers = [
        header for header in set_headers if len(header) == fewest_headers_no
    ][0]

    if max(set_lengths) != fewest_headers_no:
        print("Headers do not match:\n", set_headers)

        for headers, path in zip(set_headers, data.keys()):
            if len(headers) > fewest_headers_no:
                surplus_headers = fewest_headers.difference(headers)
                print(f"Surplus headers in {path}:\n", surplus_headers)
            else:
                missing_headers = headers.difference(fewest_headers)
                print(f"Missing headers in {path}:\n", missing_headers)

    all_unique_headers = join_headers(all_headers)

    return all_unique_headers


def get_task_result(run_result: dict[str, list], task_id: int) -> dict[str, list]:
    """
    Get the results for a task from the data.

    :param run_result: data from a result file from a specific run
    :param task_id: the task ID to select
    :return: the results for the task
    """
    indices = [i for i, x in enumerate(run_result["task_id"]) if x == task_id]
    task_results = {
        header: [value[j] for j in indices] for header, value in run_result.items()
    }
    return task_results


def join_data(
    data: dict[Path, dict[str, list]], sources_tasks: dict[Path, list[int]]
) -> tuple[list[dict], tuple]:
    """
    Join the data from the different sources into a single file with a new ID,
    formatted as a list of dictionaries for each row/part.

    :param data: data from the different result files
    :param sources_tasks: dictionary of paths and task IDs for each path
    :return: joined data
    """
    all_headers = get_headers(data)
    joined_data = []

    for path, task_ids in sources_tasks.items():
        for task_id in task_ids:
            task_results = get_task_result(run_result=data[path], task_id=task_id)
            for i in range(len(task_results["task_id"])):
                joined_data.append(
                    {header: values[i] for header, values in task_results.items()}
                )

            print(
                f"Task results for task {task_id} from {path} added to the joined data."
            )

    joined_data = sorted(joined_data, key=lambda x: x["task_id"])
    for i, row in enumerate(joined_data, start=1):
        row["id"] = i
        if not row["task"]:
            raise ValueError(f"Task is missing in row {row['id']}.", row)

    return joined_data, all_headers


def run(paths: list[str], result_directory: str) -> None:
    """
    Run the data join.

    :param paths: list of paths to the result files
    :param result_directory: path to save the joined data
    :return:
    """
    print("You are running the data join script.", end="\n\n")

    if len(paths) < 2:
        raise ValueError(
            "Please provide at least two paths to join. Now provided:", len(paths)
        )

    data = {}
    loader = DataLoader()
    data_paths = [get_paths(PREFIX / path, keyword="results") for path in paths]
    flat_paths = list(flatten(data_paths))

    for orig_path, path in zip(paths, flat_paths):
        print(f"Loading data from {orig_path}")
        data[path] = loader.load_results(path)

    id_counts = count_parts_per_task(data)
    print_counts_table(id_counts, flat_paths)

    sources_tasks = define_sources_for_tasks(id_counts, flat_paths)
    print("sources_tasks", sources_tasks)
    joined_data, headers = join_data(data, sources_tasks)

    print("\nHeaders:")
    print(*headers)

    print("\nTasks saved from each path:")
    for orig_path, task_ids in zip(paths, sources_tasks.values()):
        print("Path:", orig_path)
        print("Tasks:", *task_ids, end="\n\n")

    saver = DataSaver(save_to=PREFIX / result_directory)
    result_directory = Path(result_directory)
    (PREFIX / result_directory).mkdir(parents=True, exist_ok=True)

    saver.save_output(
        data=joined_data,
        headers=tuple(headers),
        file_name=flat_paths[0].name,
    )

    logs = [get_paths(PREFIX / path, keyword="", file_format="log") for path in paths]

    if logs:
        flat_logs = list(flatten(logs))
        log_differences = find_difference_in_paths(flat_logs)
        print("\nCopying log files found:")

        unique_logs = []
        for log, difference in zip(flat_logs, log_differences):
            unique_logs.append(
                PREFIX / result_directory / f"{log.stem}_{difference}{log.suffix}"
            )

        for path, f_log, u_log in zip(paths, flat_logs, unique_logs):
            shutil.copy2(f_log, u_log)
            if u_log.exists():
                print(f"{Path(path) / f_log.name} ==> {result_directory / u_log.name}")

            else:
                raise FileNotFoundError(f"Not copied: {Path(path) / f_log.name}")

    interpretability = [
        get_paths(PREFIX / path, keyword="interpretability", file_format="")
        for path in paths
    ]
    if interpretability:
        flat_interpretability = list(flatten(interpretability))

        attn_scores = [path / "attn_scores" for path in flat_interpretability]
        plots = [path / "plots" for path in flat_interpretability]

        for (source_path, task_ids), attn_scores_path, plots_path, path in zip(
            sources_tasks.items(), attn_scores, plots, paths
        ):
            assert str(source_path.parent) in str(attn_scores_path)

            if "before" in str(attn_scores_path):
                attn_path_add = Path("before", "interpretability")
            else:
                attn_path_add = Path("after", "interpretability")

            attn_scores_count = 0
            plots_count = 0

            for task_id in task_ids:
                attn_scores_pattern = re.compile(rf"attn_scores-{task_id}-\d+-\d+\.txt")
                x_tokens_pattern = re.compile(rf"x_tokens-{task_id}-\d+-\d+\.txt")
                y_tokens_pattern = re.compile(rf"y_tokens-{task_id}-\d+-\d+\.txt")
                attn_map_pattern = re.compile(rf"attn_map-{task_id}-\d+-\d+\.pdf")

                for attn_scores_file in attn_scores_path.iterdir():
                    if (
                        attn_scores_pattern.match(attn_scores_file.name)
                        or x_tokens_pattern.match(attn_scores_file.name)
                        or y_tokens_pattern.match(attn_scores_file.name)
                    ):
                        attn_scores_count += 1
                        dest_path = (
                            PREFIX / result_directory / attn_path_add / "attn_scores"
                        )
                        dest_path.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.copy2(
                                attn_scores_file, dest_path / attn_scores_file.name
                            )
                        except shutil.SameFileError:
                            warnings.warn(
                                f"File {attn_scores_file.name} already exists in the destination: {dest_path}"
                            )

                for plot_file in plots_path.iterdir():
                    if attn_map_pattern.match(plot_file.name):
                        plots_count += 1
                        dest_path = PREFIX / result_directory / attn_path_add / "plots"
                        dest_path.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.copy2(plot_file, dest_path / plot_file.name)
                        except shutil.SameFileError:
                            warnings.warn(
                                f"File {plot_file.name} already exists in the destination: {dest_path}"
                            )

            print(
                f"\n\nCopied interpretability files for tasks {', '.join(map(str, task_ids))} "
                f"in the following directories:"
            )
            print(
                f"{Path(path) / Path(*attn_scores_path.parts[-4:])} "
                f"==> {result_directory / attn_path_add / 'attn_scores'}\t\t({attn_scores_count} files)"
            )
            print(
                f"{Path(path) / Path(*plots_path.parts[-4:])} "
                f"==> {result_directory / attn_path_add / 'plots'}\t\t({plots_count} files)"
            )

    print(
        "\nTo obtain the accuracy of the joined data, run the evaluation script.",
        end="\n\n",
    )

    print("Data join completed.")


if __name__ == "__main__":
    # TODO: add paths of result directories that should be joined
    paths = [
        "test/test_join/20-02-2025/11-22-33/prompt_init_prompt_da_reasoning",
        "test/test_join/21-02-2025/22-33-44/prompt_init_prompt_da_reasoning",
    ]
    # TODO: path to save the joined data
    result_directory = "test/test_join/joined_data3"
    run(
        paths=paths,
        result_directory=result_directory,
    )
