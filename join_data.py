# This script is used to join the data from the different sources into a single file.
# The choice of the source file for each task is based on the number of parts:
# the file with the most parts for a task is chosen as the source for it.
# The headers are checked for matching and the unique headers are all_samples.
# The data is then all_samples and saved together in the specified directory.
# Log files are copied to there, too, but not the metrics and plots,
# because that data depends on the run being completed.

from __future__ import annotations

import re
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Generator, Iterable

from prettytable import PrettyTable

from data.DataLoader import DataLoader
from data.DataSaver import DataSaver
from plots.utils import find_difference_in_paths, get_paths

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
        elif isinstance(item, Iterable) and not isinstance(item, str):
            yield from flatten(list(item))
        else:
            yield item


def print_counts_table(
    id_counts: dict[int, list[int]], paths: list[Path], level: str = "task"
) -> None:
    """
    Print a table of counts of parts in tasks per path.
    """
    table = PrettyTable()

    # Set up the table columns: sample parts and task IDs
    table.field_names = ["Paths \\ Sample Parts"] + list(id_counts.keys())
    path_differences = find_difference_in_paths(paths)

    # Add rows for each path
    for i, name in enumerate(path_differences):
        row = [name] + [part_count[i] for part_count in id_counts.values()]
        table.add_row(row)

    print(
        f"\nCounts of{' samples' if level == 'task' else ''} parts for {level}s per path:"
    )
    print(table, end="\n\n")


def count_parts_per_level(
    data: dict[Path, dict[str, list]], level: str = "task"
) -> dict[int, list[int]]:
    """
    Count the number of parts for each task in the data.

    :param data: data from the different result files
    :param level: level of the data to count, either 'task' or 'sample'
    :return: dictionary of task IDs and counts of parts
    """
    header = "task_id"
    if level == "sample":
        header = "sample_id"

    unique_item_ids = set(flatten([result[header] for result in data.values()]))

    id_counts = defaultdict(list[int])
    for path, results in data.items():
        item_ids = results[header]
        for item_id in sorted(list(unique_item_ids)):
            id_counts[item_id].append(item_ids.count(item_id))

    return id_counts


def define_sources(
    id_counts: dict[int, list[int]], paths: list[Path], level: str = "task"
) -> dict[Path, list[int]]:
    """
    Define the source file for each task depending on the number of parts.
    The file with the most parts for a task is chosen as the source for it.

    :param id_counts: dictionary of task IDs and counts of parts
    :param paths: list of paths to the result files
    :return: dictionary of paths and task IDs for each path
    """
    items = defaultdict(list)
    for id_, counts in id_counts.items():
        max_counts = max(counts)
        if max_counts == 0:
            warnings.warn(f"No parts found for {level} {id_}.")

        inx = counts.index(max_counts)
        items[paths[inx]].append(id_)
    return items


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
        print("Headers do not match:", *set_headers, sep="\n", end="\n\n")

        all_headers = set(flatten(set_headers))
        for headers, path in zip(set_headers, data.keys()):
            if len(headers) > fewest_headers_no:
                surplus_headers = headers - fewest_headers
                warnings.warn(
                    f"* Surplus * headers in {Path(*Path(path).parts[-6:])}:\n{surplus_headers}\n\n",
                )
            if len(headers) < len(all_headers):
                missing_headers = all_headers - headers
                warnings.warn(
                    f"* Missing * headers in {Path(*Path(path).parts[-6:])}:\n{missing_headers}\n\n",
                )

    all_unique_headers = join_headers(all_headers)

    return all_unique_headers


def get_level_result(
    run_result: dict[str, list], task_id: int, header: str
) -> dict[str, list]:
    """
    Get the results for a task from the data.

    :param run_result: data from a result file from a specific run
    :param task_id: the task ID to select
    :param header: the header to select
    :return: the results for the task
    """
    indices = [i for i, x in enumerate(run_result[header]) if x == task_id]
    task_results = {
        header: [value[j] for j in indices] for header, value in run_result.items()
    }
    return task_results


def join_data(
    data: dict[Path, dict[str, list]],
    sources_item_ids: dict[Path, list[int]],
    level: str = "task",
) -> tuple[list[dict], tuple]:
    """
    Join the data from the different sources into a single file with a new ID,
    formatted as a list of dictionaries for each row/part.

    :param data: data from the different result files
    :param sources_item_ids: dictionary of paths and item IDs for each path
    :param level: level of the data to join, either 'task' or 'sample'
    :return: all_samples data
    """
    header = "task_id"
    if level == "sample":
        header = "sample_id"
    all_headers = get_headers(data)
    joined_data = []
    all_item_ids = sorted(map(int, flatten(list(sources_item_ids.values()))))

    for item_id in all_item_ids:
        if max(sources_item_ids.values()) == 0:
            warnings.warn(
                f"{level.capitalize()} {sources_item_ids} not found in the sources."
            )
            continue
        for path, task_ids in sources_item_ids.items():
            if item_id in task_ids:
                task_results = get_level_result(
                    run_result=data[path], task_id=item_id, header=header
                )
                for i in range(len(task_results[header])):
                    joined_data.append(
                        {header: values[i] for header, values in task_results.items()}
                    )

                print(
                    f"{level.capitalize()} results for {level} {item_id} "
                    f"from {Path(*Path(path).parts[-4:])} were added to the all_samples data."
                )

    joined_data = sorted(joined_data, key=lambda x: x[header])
    filtered_joined_data = []
    i = 1
    for row in joined_data:
        if type(row["id_"]) is str and not row["id_"].isdigit():
            continue
        if not row["task"]:
            warnings.warn(
                f"{level.capitalize()} is missing in row {row['id_']}:\n{row}"
            )
            continue
        row["id_"] = i
        i += 1
        filtered_joined_data.append(row)

    return filtered_joined_data, all_headers


def run(
    paths: list[str],
    result_directory: str,
    level: str = "task",
    keyword: str = "results",
) -> None:
    """
    Run the data join.

    :param paths: list of paths to the result files
    :param result_directory: path to save the all_samples data
    :param level: level of the data to join, either 'task' or 'sample'
    :param keyword: type_ to search for in the paths
    :return:
    """
    print("You are running the data join script.", end="\n\n")

    if level not in ["task", "sample"]:
        raise ValueError(
            f"Level '{level}' not recognized. Please choose 'task' or 'sample''."
        )

    if len(paths) < 2:
        raise ValueError(
            "Please provide at least two paths to join. Now provided:", len(paths)
        )

    data = {}
    loader = DataLoader()
    data_paths = [get_paths(PREFIX / path, keyword=keyword) for path in paths]
    flat_paths = list(flatten(data_paths))

    for path in flat_paths:
        print(f"Loading data from {Path(*Path(path).parts[-6:])}")
        data[path] = loader.load_results(path)

    print("Number of files:", len(data))

    id_counts = count_parts_per_level(data, level)
    print_counts_table(id_counts, flat_paths, level)

    sources_items = define_sources(id_counts, flat_paths)
    joined_data, headers = join_data(data, sources_items, level)

    print("\nHeaders:")
    print(*headers)

    print(f"\n{level.capitalize()}s saved from each path:")
    for path, item_ids in zip(flat_paths, sources_items.values()):
        print("Path:", Path(*path.parts[-4:]))
        print(f"{level.capitalize()}s:", *item_ids, end="\n\n")

    full_result_directory = PREFIX / result_directory
    full_result_directory.mkdir(parents=True, exist_ok=True)
    if next(full_result_directory.iterdir(), None):
        raise FileExistsError(
            f"Directory {result_directory} is not empty. Please provide an empty directory."
        )

    saver = DataSaver(save_to=full_result_directory)
    saver.save_output(
        data=joined_data,
        headers=tuple(headers),
        file_name=f"joined_{level}_results.csv",
    )

    logs = [get_paths(PREFIX / path, keyword="", file_format="log") for path in paths]

    if logs:
        flat_logs = list(flatten(logs))
        log_differences = find_difference_in_paths(flat_logs)
        if "" in log_differences:
            raise NameError(
                "The structure or log file names is not uniform. Please add distinctions to standard log files."
            )

        print("\nCopying log files found:")

        unique_logs = []
        for log, difference in zip(flat_logs, log_differences):
            if difference != log.stem:
                unique_logs.append(
                    full_result_directory / f"{log.stem}_{difference}{log.suffix}"
                )
            else:
                unique_logs.append(full_result_directory / log.name)

        for f_log, u_log in zip(flat_logs, unique_logs):
            shutil.copy2(f_log, u_log)
            if u_log.exists():
                print(
                    f"{Path(*f_log.parts[-4:])} ==> {Path(result_directory) / u_log.name}"
                )
            else:
                raise FileNotFoundError(f"Not copied: {Path(*f_log.parts[-4:])}")

    interpretability = [
        get_paths(PREFIX / path.parent, keyword="interpretability", file_format="")
        for path in flat_paths
    ]
    if interpretability:
        flat_interpretability = list(flatten(interpretability))

        attn_scores = [path / "attn_scores" for path in flat_interpretability]
        plots = [path / "plots" for path in flat_interpretability]

        if not (
            len(attn_scores) == len(plots) == len(flat_paths) == len(sources_items)
        ):
            raise ValueError(
                "The number of paths for attention scores and plots do not match."
            )
        for (source_path, item_ids), attn_scores_path, plots_path, path in zip(
            sources_items.items(), attn_scores, plots, flat_paths
        ):
            assert str(source_path.parent.name) in str(
                attn_scores_path
            ), f"Path {source_path.parent.name} not found in {attn_scores_path}"

            if "before" in str(attn_scores_path):
                attn_path_add = Path("before", "interpretability")
            else:
                attn_path_add = Path("after", "interpretability")

            attn_scores_count = 0
            plots_count = 0

            for item_id in item_ids:
                (
                    attn_scores_pattern,
                    x_tokens_pattern,
                    y_tokens_pattern,
                    attn_map_pattern,
                ) = (
                    None,
                    None,
                    None,
                    None,
                )
                if level == "task":
                    attn_scores_pattern = re.compile(
                        rf"attn_scores-{item_id}-\d+-\d+\.txt"
                    )
                    x_tokens_pattern = re.compile(rf"x_tokens-{item_id}-\d+-\d+\.txt")
                    y_tokens_pattern = re.compile(rf"y_tokens-{item_id}-\d+-\d+\.txt")
                    attn_map_pattern = re.compile(rf"attn_map-{item_id}-\d+-\d+\.pdf")
                elif level == "sample":
                    attn_scores_pattern = re.compile(
                        rf"attn_scores-\d+-{item_id}-\d+\.txt"
                    )
                    x_tokens_pattern = re.compile(rf"x_tokens-\d+-{item_id}-\d+\.txt")
                    y_tokens_pattern = re.compile(rf"y_tokens-\d+-{item_id}-\d+\.txt")
                    attn_map_pattern = re.compile(rf"attn_map-\d+-{item_id}-\d+\.pdf")

                for attn_scores_file in attn_scores_path.iterdir():
                    if (
                        attn_scores_pattern.match(attn_scores_file.name)
                        or x_tokens_pattern.match(attn_scores_file.name)
                        or y_tokens_pattern.match(attn_scores_file.name)
                    ):
                        attn_scores_count += 1
                        dest_path = (
                            full_result_directory / attn_path_add / "attn_scores"
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
                if attn_scores_count == 0:
                    warnings.warn(
                        f"No attention scores found for {item_id} in {attn_scores_path}"
                    )

                if plots_path.exists():
                    for plot_file in plots_path.iterdir():
                        if attn_map_pattern.match(plot_file.name):
                            plots_count += 1
                            dest_path = (
                                PREFIX / result_directory / attn_path_add / "plots"
                            )
                            dest_path.mkdir(parents=True, exist_ok=True)
                            try:
                                shutil.copy2(plot_file, dest_path / plot_file.name)
                            except shutil.SameFileError:
                                warnings.warn(
                                    f"File {plot_file.name} already exists in the destination: {dest_path}"
                                )
                else:
                    warnings.warn(f"No plots found in {plots_path}")

            print(
                f"\n\nCopied interpretability files for {level} {', '.join(map(str, item_ids))} "
                f"in the following directories:"
            )
            print(
                f"{Path(*Path(path, attn_scores_path).parts[-6:])} "
                f"==> {Path(*Path(result_directory / attn_path_add / 'attn_scores').parts[-6:])}"
                f"\t\t({attn_scores_count} files)"
            )
            print(
                f"{Path(*Path(path, plots_path).parts[-6:])} "
                f"==> {Path(*Path(result_directory / attn_path_add / 'plots').parts[-6:])}"
                f"\t\t({plots_count} files)"
            )

    print(
        "\nTo obtain the accuracy of the all_samples data, run the evaluation script.",
        end="\n\n",
    )

    print("Data join completed successfully.")


if __name__ == "__main__":
    # TODO: add paths of result directories that should be all_samples
    paths = [
        "results/skyline/valid/with_examples/reasoning/tasks_1_2",
        "results/skyline/valid/with_examples/reasoning/tasks_3/all_samples",
        "results/skyline/valid/with_examples/reasoning/tasks_4_10",
        "results/skyline/valid/with_examples/reasoning/tasks_11_20",
    ]
    # TODO: path to save the all_samples data
    result_directory = "results/skyline/valid/with_examples/reasoning/all_tasks"
    run(paths=paths, result_directory=result_directory, level="task")
