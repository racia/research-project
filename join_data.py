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
    all_unique_headers = set(flatten(set_headers))

    if max(set_lengths) != fewest_headers_no:
        warnings.warn("Headers do not match!")

    return tuple(all_unique_headers)


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


def copy_folder_files(
    source_path: Path, dest_path: Path, filter_pattern: re.Pattern = None
) -> None:
    """
    Copy interpretability files from the source path to the destination path.
    Does not disambiguate the source paths, so the files should be unique.
    Filtering is only applied to files, not directories.
    """
    dest_path = dest_path / source_path.name
    path_counter = 0
    dest_path.mkdir(parents=True, exist_ok=True)
    for path in source_path.iterdir():
        if path.is_dir():
            copy_folder_files(path, dest_path, filter_pattern)
        elif path.is_file():
            if "metrics" in path.name:
                continue
            if filter_pattern and not filter_pattern.search(path.name):
                continue
            try:
                shutil.copy2(path, dest_path / path.name)
                path_counter += 1
            except shutil.SameFileError:
                warnings.warn(
                    f"File '{path.name}' already exists in the destination: {dest_path}"
                )
        else:
            warnings.warn(f"'{path}' is not a not a directory, nor a file. Skipping..")
            continue
    print(
        f"{Path(*Path(source_path).parts[-4:])} ==> {dest_path} ({path_counter} files copied)"
    )


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
    source_paths: list[str],
    target_directory: str,
    level: str = "task",
    keyword: str = "results",
) -> None:
    """
    Run the data join.

    :param source_paths: list of paths to the result files to move
    :param target_directory: path to save the all_samples data
    :param level: level of the data to join, either 'task' or 'sample'
    :param keyword: type_ to search for in the paths
    :return:
    """
    print("You are running the data join script.", end="\n\n")

    if level not in ["task", "sample"]:
        raise ValueError(
            f"Level '{level}' not recognized. Please choose 'task' or 'sample''."
        )

    if len(source_paths) < 2:
        raise ValueError(
            "Please provide at least two source_paths to join. Now provided:",
            len(source_paths),
        )

    full_result_directory = PREFIX / target_directory
    full_result_directory.mkdir(parents=True, exist_ok=True)
    if next(full_result_directory.iterdir(), None):
        raise FileExistsError(
            f"Directory {target_directory} is not empty. Please provide an empty directory."
        )

    data = {}
    loader = DataLoader()
    data_paths = [get_paths(PREFIX / path, keyword=keyword) for path in source_paths]
    flat_paths = list(flatten(data_paths))

    for path in flat_paths:
        results, _ = loader.load_results(path, list_output=False)
        if not results:
            warnings.warn(
                f"No data found in {Path(*Path(path).parts[-6:])}. Skipping this path."
            )
            continue
        print(f"Loaded data from {Path(*Path(path).parts[-6:])}")
        data[path] = results

    print("Number of files:", len(data))

    id_counts = count_parts_per_level(data, level)
    print_counts_table(id_counts, flat_paths, level)

    sources_items = define_sources(id_counts, flat_paths)
    joined_data, headers = join_data(data, sources_items, level)

    print("\nHeaders:")
    print(*headers)

    saver = DataSaver(save_to=full_result_directory, loaded_baseline_results=False)
    saver.save_output(
        data=joined_data,
        headers=tuple(headers),
        file_name=f"joined_{keyword}_{level}_results.csv",
    )

    folders_to_move = ["iterations", "sample_results", "before", "after"]
    source_paths = set(
        [
            path.parent if level == "task" else path.parent.parent
            for path in sources_items.keys()
        ]
    )
    differences = find_difference_in_paths(list(source_paths))
    ids = "|".join(map(str, flatten(list(sources_items.values()))))
    assert len(differences) == len(source_paths), (
        f"The number of differences in the source paths does not match the number of source paths: "
        f"{len(differences)} != {len(source_paths)}."
    )
    if re.search(r"\d+", keyword):
        key = re.search(r"\d+", keyword).group(0)
    else:
        key = r"\d+"
    for source_path, diff in zip(source_paths, differences):
        path_counter = 0
        if not diff:
            diff = f"path_{path_counter}"
        if level == "task":
            filter_pattern = re.compile(rf"(?:{ids})-\d+-\d+|t_(?:{ids})_s_\d+")
        elif level == "sample":
            filter_pattern = re.compile(rf"{key}-(?:{ids})-\d+|t_{key}_s_(?:{ids})")
        else:
            filter_pattern = re.compile(r"")
        for path in Path(source_path).iterdir():
            if path.is_file() and "results" in path.name and path.name.endswith(".csv"):
                # skip the results files, they are already joined and saved
                continue
            if path.is_dir() and path.name in folders_to_move:
                copy_folder_files(path, full_result_directory, filter_pattern)
            if path.is_dir() and path.name.startswith("."):
                file_name = f"{path.stem}_{diff}{path.suffix}"
                copy_folder_files(
                    path, full_result_directory / file_name, filter_pattern
                )
            if path.is_file():
                try:
                    file_name = f"{path.stem}_{diff}{path.suffix}"
                    shutil.copy2(path, full_result_directory / file_name)
                    path_counter += 1
                except shutil.SameFileError:
                    warnings.warn(
                        f"File '{path.name}' already exists in the destination: {target_directory}"
                    )
        print(
            f"{Path(*Path(source_path).parts[-4:])} ==> {Path(target_directory)} ({path_counter} files copied)"
        )

    print("\nData join completed successfully.")

    print(
        "To obtain the accuracy of the all_samples data, run the evaluation script.",
        end="\n\n",
    )


if __name__ == "__main__":
    paths = [
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_1",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_2",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_3",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/16-10-2025/task_4/task_4",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_5",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_6",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_7",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_8",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_9",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_10",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_11",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_12",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_13",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_14",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/19-10-2025/task_15/task_15",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/15-09-2025/task_16/task_16",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_17",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_18",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/16-10-2025/task_19/task_19",
        "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/task_20",
    ]

    result_directory = "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/all_tasks_joined"
    run(
        source_paths=paths,
        target_directory=result_directory,
        level="task",
        keyword="_results",  # example: "t_20" for a specific task, "reasoning_results" for generally saved results
    )  # might not work if too general!
