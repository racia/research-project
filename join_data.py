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


FOLDERS_TO_MOVE = ["iterations", "sample_results", "before", "after"]


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
    :param level: level of the data to count, either 'task' or 'sample'
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
    path_counter = 0
    dest_path.mkdir(parents=True, exist_ok=True)
    for path in source_path.iterdir():
        if path.is_dir():
            copy_folder_files(path, dest_path / path.name, filter_pattern)
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


def process_path(
    path, diff, full_result_directory, filter_pattern, target_directory, path_counter
):
    print("Inspecting path:", path)
    if path.is_file() and "results" in path.name and path.name.endswith(".csv"):
        # skip the results files, they are already joined and saved
        return path_counter
    elif path.is_dir() and path.name in FOLDERS_TO_MOVE:
        copy_folder_files(path, full_result_directory / path.name, filter_pattern)
        return path_counter
    elif path.is_dir() and path.name.startswith("."):  # hidden folders
        file_name = f"{path.stem}_{diff}{path.suffix}"
        copy_folder_files(path, full_result_directory / file_name)
        return path_counter
    elif path.is_file():
        try:
            file_name = f"{path.stem}_{diff}{path.suffix}"
            shutil.copy2(path, full_result_directory / file_name)
            path_counter += 1
        except shutil.SameFileError:
            warnings.warn(
                f"File '{path.name}' already exists in the destination: {target_directory}"
            )
        return path_counter
    else:
        print("Going one level deeper...")
        for path in path.iterdir():
            path_counter = process_path(
                path,
                diff,
                full_result_directory,
                filter_pattern,
                target_directory,
                path_counter,
            )
        return path_counter


def run(
    source_paths: list[str],
    target_directory: str,
    level: str = "task",
    keyword: str = "results",
    task: str = "evaluation",
) -> None:
    """
    Run the data join.

    :param source_paths: list of paths to the result files to move
    :param target_directory: path to save the all_samples data
    :param level: level of the data to join, either 'task' or 'sample'
    :param keyword: type_ to search for in the paths
    :param task: task type, either 'reasoning' or 'direct_answer', used in the results file name
    :return: None
    """
    print("You are running the data joining script.", end="\n\n")

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
    additions = [keyword.strip("_"), f"{task}_results"]
    saver.save_output(
        data=joined_data,
        headers=tuple(headers),
        file_name=f"joined_{additions[0] if additions[0]==additions[1] else '_'.join(additions)}.csv",
    )

    source_paths = set(
        [
            path.parent if level == "task" else path.parent.parent
            for path in sources_items.keys()
        ]
    )
    differences = find_difference_in_paths(list(source_paths))
    trimmed_source_paths = []
    for path in source_paths:
        while path.name not in differences:
            path = path.parent
        trimmed_source_paths.append(path)

    print(
        "\nFound the following differences in the paths:",
        *differences,
        sep="\n- ",
        end="\n\n",
    )
    assert len(differences) == len(trimmed_source_paths), (
        f"The number of differences in the source paths does not match the number of source paths: "
        f"{len(differences)} != {len(trimmed_source_paths)}."
    )
    ids = "|".join(map(str, flatten(list(sources_items.values()))))
    if re.search(r"\d+", keyword):
        key = re.search(r"\d+", keyword).group(0)
    else:
        key = r"\d+"
    for source_path, diff in zip(trimmed_source_paths, differences):
        path_counter = 0
        if not diff:
            warnings.warn("Path diff is empty string for", source_path)
            diff = f"path_{path_counter}"
        if level == "task":
            filter_pattern = re.compile(rf"[-_](?:{ids})-\d+-\d+|t_(?:{ids})_s_\d+")
        elif level == "sample":
            filter_pattern = re.compile(rf"[-_]{key}-(?:{ids})-\d+|t_{key}_s_(?:{ids})")
        else:
            filter_pattern = re.compile(r"")
        for path in Path(source_path).iterdir():
            path_counter = process_path(
                path,
                diff,
                full_result_directory,
                filter_pattern,
                target_directory,
                path_counter,
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
    baseline_da_v1_t_20 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v1/tasks_16_19_full_task_20_s_1_91",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v1/task_20_s_91_93",
    ]
    baseline_da_v1 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v1/tasks_1_2",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v1/tasks_3_15",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v1/tasks_16_19_full_task_20_s_1_91",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v1/task_20",
    ]
    baseline_da_v2 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v2/tasks_1_2",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v2/tasks_3_15",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v2/tasks_16_20",
    ]
    baseline_da_v3 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v3/tasks_1_5",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v3/task_6",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v3/tasks_7_12_full_13_s_1_30",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v3/tasks_13_20",
    ]
    baseline_da_v4 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v4/tasks_1_5",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v4/tasks_6_10",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v4/tasks_11_15",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v4/tasks_16_20",
    ]
    baseline_da_v5 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v5/tasks_1_5",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v5/tasks_3_15",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/da/v5/tasks_16_20",
    ]
    baseline_reasoning_v1_t_20 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v1/tasks_16_20",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v1/task_20_s_90_93",
    ]
    baseline_reasoning_v1 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v1/tasks_1_5",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v1/tasks_6_10",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v1/tasks_11_15",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v1/tasks_16_20",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v1/task_20",
    ]
    baseline_reasoning_v2_t_12 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v2/tasks_1_11_full_12_s_1_33",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v2/task_12_s_33_100_tasks_13_20_full",
    ]
    baseline_reasoning_v2 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v2/tasks_1_11_full_12_s_1_33",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v2/task_12",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v2/task_12_s_33_100_tasks_13_20_full",
    ]
    baseline_reasoning_v3_t_12 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v3/tasks_1_11_full_12_s_1_41",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v3/task_12_s_33_100_tasks_13_19_full_task_20_s_1_72",
    ]
    baseline_reasoning_v3_t_20 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v3/task_12_s_33_100_tasks_13_19_full_task_20_s_1_72",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v3/task_20_s_72_93",
    ]
    baseline_reasoning_v3 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v3/tasks_1_11_full_12_s_1_41",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v3/task_12",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v3/task_12_s_33_100_tasks_13_19_full_task_20_s_1_72",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v3/task_20",
    ]
    baseline_reasoning_v4_t_3_19 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v4/task_3_19_s_1_68",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v4/tasks_3_19_s_68_100",
    ]
    baseline_reasoning_v4 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v4/tasks_1_2_full_3_s_1_68",
        *[
            f"/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v4/task_{i}"
            for i in range(3, 21)
        ],
    ]
    baseline_reasoning_v5_t_11 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v5/task_11_s_1_5",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v5/tasks_10_11_s_5_100",
    ]
    baseline_reasoning_v5_t_12_20 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v5/tasks_12_20_s_1_33",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v5/tasks_12_20_s_33_100",
    ]
    baseline_reasoning_v5 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v5/tasks_1_2_full_3_s_1_15",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v5/task_3",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v5/tasks_3_s_15_100_t_4_9_full_t_10_s_1_4",
        *[
            f"/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v5/task_{i}"
            for i in range(10, 21)
        ],
    ]
    skyline_da_v1 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v1/tasks_1_2",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v1/tasks_3_5",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v1/tasks_6_7_full_8_s_1_77",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v1/tasks_8_10",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v1/tasks_11_15",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v1/tasks_16_20",
    ]
    skyline_da_v2_t_16_20 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v2/unjoined/tasks_16_20_s_1_65",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v2/unjoined/tasks_16_20_s_66_100",
    ]
    skyline_da_v2 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v2/tasks_1_2",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v2/tasks_3_5",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v2/tasks_6_7",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v2/tasks_8_10",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v2/tasks_11_15",
        *[
            f"/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v2/task_{i}"
            for i in range(16, 21)
        ],
    ]
    skyline_da_v3_t_3 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v3/unjoined/task_3_s_1_27",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v3/unjoined/task_3_s_25_100",
    ]
    skyline_da_v3 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v3/tasks_1_2",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v3/task_3",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v3/tasks_4_10",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v3/tasks_11_15",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v3/tasks_16_20",
    ]
    skyline_da_v4_t_3 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v4/unjoined/task_3_s_1_22",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v4/unjoined/task_3_s_23_100",
    ]
    skyline_da_v4_t_14 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v4/tasks_4_13_full_14_s_1_26",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v4/task_14_s_25_100_task_15_full",
    ]
    skyline_da_v4 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v4/tasks_1_2",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v4/tasks_4_13_full_14_s_1_26",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v4/task_14_s_25_100_task_15_full",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v4/tasks_16_20",
    ]
    skyline_da_v5_t_3 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/unjoined/task_3_s_1_28",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/unjoined/task_3_s_23_99",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/unjoined/task_3_s_99_100",
    ]
    skyline_da_v5_t_14 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/tasks_4_13_full_task_14_s_1_49",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/task_14_s_49_100_task_15_full",
    ]
    skyline_da_v5_t_20 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/unjoined/task_20_s_1_15",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/unjoined/task_20_s_15_69",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/unjoined/task_20_s_69_89",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/unjoined/task_20_s_89_93",
    ]
    skyline_da_v5 = [
        # Found the following differences in the paths:
        # - tasks_4_13_full_task_14_s_1_49
        # - task_2
        # - task_20
        # - tasks_16_19
        # - task_14_s_49_100_task_15_full
        # - task_14
        # - task_3
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/task_1",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/task_2",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/task_3",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/tasks_4_13_full_task_14_s_1_49",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/task_14",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/task_14_s_49_100_task_15_full",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/tasks_16_19",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/da/v5/task_20",
    ]
    skyline_reasoning = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/reasoning/v1/tasks_1_2",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/reasoning/v1/tasks_3_5",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/reasoning/v1/tasks_6_9",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/reasoning/v1/task_10",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/reasoning/v1/tasks_11_14",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/reasoning/v1/task_15",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/reasoning/v1/tasks_16_19",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/reasoning/v1/task_20",
    ]
    sd_reasoning_v1_t_1 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_1/task_1_s_1_84",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_1/task_1_s_84_100",
    ]
    sd_reasoning_v1_t_2 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_2/task_2_s_1_76",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_2/task_2_s_76_100",
    ]
    sd_reasoning_v1_t_3 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_3/task_3_s_1_51",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_3/task_3_s_51_59",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_3/task_3_s_59_86",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_3/task_3_s_86_100",
    ]
    sd_reasoning_v1_t_5 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_5/task_5_s_61_89",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_5/task_5_s_89_100",
    ]
    sd_reasoning_v1_t_6 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_6/task_6_s_1_21",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_6/task_6_s_21_100",
    ]
    sd_reasoning_v1_t_7 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_7/task_7_s_1_92",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_7/task_7_s_92_100",
    ]
    sd_reasoning_v1_t_8 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_8/task_8_s_1_59",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_8/task_8_s_59",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_8/task_8_s_60_100",
    ]
    sd_reasoning_v1_t_9 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_9/task_9_s_1_78",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_9/task_9_s_78_83",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_9/task_9_s_83_100",
    ]
    sd_reasoning_v1_t_10 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_10/task_10_s_1_88",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_10/task_10_s_88_100",
    ]
    sd_reasoning_v1_t_11 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_11/task_11_s_1_89",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_11/task_11_s_89_100",
    ]
    sd_reasoning_v1_t_12 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_12/task_12_s_1_93",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_12/task_12_s_93_100",
    ]
    sd_reasoning_v1_t_13 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_13/task_13_s_1_89",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_13/task_13_s_89_93",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_13/task_13_s_93_100",
    ]
    sd_reasoning_v1_t_14 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_14/task_14_s_1_60",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_14/task_14_s_59_100",
    ]
    sd_reasoning_v1_t_17 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_17/task_17_s_1_43",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_17/task_17_s_43_84",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_17/task_17_s_84_97",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_17/task_17_s_97_100",
    ]
    sd_reasoning_v1_t_18 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_18/task_18_s_1_84",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_18/task_18_s_84_100",
    ]
    # TODO: unfinished!
    sd_reasoning_v1_t_20 = [
        # samples 1-27 unfinished
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_20/task_20_s_27_62",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/unjoined/task_20/task_20_s_62_91",
        # samples 92-93 unfinished
    ]
    sd_reasoning = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_1",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_2",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_3",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning//task_4",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_5",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_6",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_7",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_8",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_9",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_10",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_11",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_12",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_13",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_14",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_15",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_16",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_17",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_18",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_19",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/task_20",
    ]
    feedback_reasoning_v1_t_3 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/task_3/t_3_s_1_9",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/task_3/t_3_s_10_32",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/task_3/t_3_s_32_56",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/task_3/t_3_s_56_77",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/task_3/t_3_s_77_96",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/task_3/t_3_s_96_100",
    ]
    feedback_reasoning_v1_t_4 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/task_4/t_4_s_1_63",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/task_4/t_4_s_64_100",
    ]
    feedback_reasoning_v1_t_5 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/task_5/t_5_s_1_33__s_33_unfinished",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/task_5/t_5_s_56_79__s_79_unfinished",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/task_5/t_5_s_34_56_s_80_89",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/task_5/t_5_s_90_100",
    ]
    feedback_reasoning_v1_t_20 = [
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/tasks_19_20/t_20_1_15",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/tasks_19_20/t_20_s_14_67",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/unjoined/tasks_19_20/t_19_20_s_66_100_v1",
    ]
    feedback_reasoning_v1 = [
        *[
            f"/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/task_{i}"
            for i in range(1, 8)
        ],
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/tasks_8_10",
        *[
            f"/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/task_{i}"
            for i in range(11, 17)
        ],
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/tasks_17|19",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/task_18",
        "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v1/task_20",
    ]
    # TODO: NB! The difference in paths the script should detect must be on the same level in the file tree!
    paths = []
    result = f"/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/task_9"
    run(
        source_paths=sd_reasoning_v1_t_9,
        target_directory=result,
        level="sample",  # 'task' or 'sample'
        # might not work if too general! try "_results"
        keyword=f"t_9",  # example: "t_20" for a specific task,
        # "reasoning_results", "direct_answer_results", for generally saved results
        task="reasoning",  # 'reasoning' or 'direct_answer' (direct answer)
    )
    # TODO: once finished, check the attention scores! They all should be saved in the same folder
