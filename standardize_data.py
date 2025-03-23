# Description: Standardize the data for the evaluation pipeline.
# 1) Load the data from the path.
# 2) Add missing columns: part id, silver reasoning.
# 3) Add part IDs to the data.
# 4) Save the data.

import re
import warnings
from pathlib import Path

from data.DataLoader import DataLoader
from data.DataSaver import DataSaver
from inference.DataLevels import Results, SamplePart


def extract_split(data_path: str) -> str:
    """
    Extract the split from the data path. If the split is not found, return "split".

    :param data_path: The path to the data.
    :return: The split.
    """
    for split in ["valid", "test", "train"]:
        if split in data_path:
            return split
    return "split"


def get_silver_reasoning(
    task_id: int,
    sample_id: int,
    part_id: int,
    silver_reasoning_data: dict[tuple[int, int, int], dict],
    from_zero: bool = False,
) -> str:
    """
    Get the silver reasoning for the part.

    :param task_id: The task id.
    :param sample_id: The sample id.
    :param part_id: The part id.
    :param silver_reasoning_data: The silver reasoning data.
    :param from_zero: Whether the part ids start from zero.
    """
    if from_zero:
        sample_id -= 1
        part_id -= 1

    if (task_id, sample_id, part_id) in silver_reasoning_data:
        return silver_reasoning_data[(task_id, sample_id, part_id)]["reasoning"]
    raise ValueError(
        f"Silver reasoning for <task_id='{task_id}', sample_id='{sample_id}', part_id='{part_id}'> not found."
    )


def add_part_ids(parts: list[SamplePart]) -> list[SamplePart]:
    """
    Add part ids to the parts.

    :param parts: The parts to add the path ids to.
    :return: The parts with the path ids.
    """
    print("Adding part IDs to the data...")
    part_id = 1
    for inx, part in enumerate(parts):
        if inx == 0 or parts[inx - 1].sample_id != part.sample_id:
            part_id = 1
        if part.part_id == 0:
            part.part_id = part_id
            part_id += 1
        else:
            raise ValueError("Part IDs are already present.")
    return parts


def run(
    data_path: str,
    headers: dict[str, list[str]],
    add_silver_reasoning: bool = False,
) -> None:
    """
    Run the standardization pipeline.

    :param data_path: Path to the data to standardize
    :param headers: The headers for the data
    :param add_silver_reasoning: Whether to include silver reasoning data
    :return: None
    """
    loader = DataLoader()
    saver = DataSaver(save_to=str(Path(data_path).parent))

    headers_results_before = [f"{result}_before" for result in headers["results"]]
    headers_results_after = [f"{result}_after" for result in headers["results"]]
    all_headers = (
        headers["general"]
        + headers_results_before
        + headers["results"]
        + headers_results_after
    )
    print("Using long list of headers to load the data.")

    data = loader.load_result_data(data_path, headers=all_headers, list_output=True)

    silver_reasoning_data = None
    if add_silver_reasoning:
        silver_reasoning_path = "data/silver_reasoning_test.csv"
        silver_reasoning_headers = [
            "task_id",
            "sample_id",
            "part_id",
            "context",
            "question",
            "answer",
            "reasoning",
        ]
        silver_reasoning_data = loader.load_reasoning_data(
            path=silver_reasoning_path,
            headers=silver_reasoning_headers,
        )

    parts = []
    counter = 0
    for row in data:
        assert type(row) == dict

        if not row["task"]:
            warnings.warn(
                f"Data is missing for task {row['task_id']}, skipping the row."
            )
            continue

        counter += 1

        # Add missing columns
        row["part_id"] = 0 if "part_id" not in row else row["part_id"]
        row["model_output"] = "" if "model_output" not in row else row["model_output"]
        row["model_reasoning"] = (
            "" if "model_reasoning" not in row else row["model_reasoning"]
        )
        row["silver_reasoning"] = (
            "" if "silver_reasoning" not in row else row["silver_reasoning"]
        )
        h_patt = re.compile(r"(.+)_(?:after|before)")
        row["id_"] = counter

        print(
            "\n".join(
                f"{gen_header}: {row[gen_header]}" for gen_header in headers["general"]
            ),
            end="\n\n",
        )
        part = SamplePart(
            **{gen_header: row[gen_header] for gen_header in headers["general"]},
            multi_system=True,
        )
        if "model_answer_before" in row and row["model_answer_before"]:
            part.result_before = Results(
                **dict(
                    [
                        (
                            (h_patt.match(result)[1], str(row[result]))
                            if h_patt.match(result)
                            else (result, row[result])
                        )
                        for result in headers_results_before
                    ]
                ),
                after=False,
            )
        print(
            "\n".join(
                f"{h_patt.match(result)[1] if h_patt.match(result) else result}: {row[result]}"
                for result in headers_results_after
            ),
            end="\n\n",
        )
        if "model_answer_after" in row:
            part.result_after = Results(
                **dict(
                    [
                        (
                            (h_patt.match(result)[1], str(row[result]))
                            if h_patt.match(result)
                            else (result, row[result])
                        )
                        for result in headers_results_after
                    ]
                ),
                after=True,
            )
        else:
            part.result_after = Results(
                **dict(
                    [
                        (
                            (h_patt.match(result)[1], str(row[result]))
                            if h_patt.match(result)
                            else (result, row[result])
                        )
                        for result in headers["results"]
                    ]
                ),
                after=True,
            )

        parts.append(part)

    if parts[0].part_id == 0:
        print("Part IDs are missing in the data.")
        parts = add_part_ids(parts)

    if add_silver_reasoning:
        if not silver_reasoning_data:
            raise ValueError("Silver reasoning data is not loaded.")
        print("Adding silver reasoning to the data...")
        for part in parts:
            part.silver_reasoning = get_silver_reasoning(
                task_id=part.task_id,
                sample_id=part.sample_id,
                part_id=part.part_id,
                silver_reasoning_data=silver_reasoning_data,
                from_zero=True,
            )

    print("Saving the data...")

    path = Path(data_path)
    file_path = path.parent / f"{path.stem}_upd.csv"

    part_dicts = [part.get_result() for part in parts]
    part_headers = list(part_dicts[0].keys())

    saver.save_output(part_dicts, part_headers, file_path, flag="w")
    print(f"Saved the data with part IDs to {file_path}.")


if __name__ == "__main__":
    # TODO: Add the link to the data
    data_path = ""
    # TODO: set to True if reasoning is present in the data
    add_silver_reasoning = False
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
        "results": [  # for both before and after
            "model_answer",
            "model_reasoning",
            "model_output",  # TODO: make sure it's not 'model_result'
            "correct",
        ],
    }
    run(
        data_path=data_path,
        headers=headers,
        add_silver_reasoning=add_silver_reasoning,
    )
