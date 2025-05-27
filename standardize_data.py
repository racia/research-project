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
from evaluate_data import SilverReasoning
from inference.DataLevels import Results, SamplePart


def add_part_ids(parts: list[SamplePart]) -> list[SamplePart]:
    """
    Add part ids to the parts.

    :param parts: The parts to addition the path ids to.
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


def run(data_path: str) -> None:
    """
    Run the standardization pipeline.

    :param data_path: Path to the data to standardize
    :return: None
    """
    if not data_path:
        raise ValueError("Data path is missing.")

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
    loader = DataLoader()
    saver = DataSaver(save_to=str(Path(data_path).parent))

    headers_results_before = [f"{result}_before" for result in headers["results"]]
    headers_results_after = [f"{result}_after" for result in headers["results"]]
    print("Using long list of headers to load the data.")

    data, _ = loader.load_results(data_path, list_output=True)
    silver_reasoning = SilverReasoning(loader)

    header_mapping = {
        "true_result": "golden_answer",
        "model_result": "model_output",
        "sample_no": "sample_id",
        "id": "id_",
        "correct?": "correct",
    }
    multi_system = False
    for header, header_upd in header_mapping.items():
        for row in data:
            if header in row:
                row[header_upd] = row.pop(header)
            if "model_answer_before" in row:
                multi_system = True

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
            multi_system=multi_system,
        )
        if "model_answer_before" in row and row["model_answer_before"]:
            part.results.append(
                Results(
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
                    version="before",
                )
            )

        if "model_answer_after" in row and row["model_answer_after"]:
            print(
                "\n".join(
                    f"{h_patt.match(result)[1] if h_patt.match(result) else result}: {row[result]}"
                    for result in headers_results_after
                ),
                end="\n\n",
            )
            part.results.append(
                Results(
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
                    version="after",
                )
            )
        else:
            print(
                "\n".join(
                    f"{h_patt.match(result)[1] if h_patt.match(result) else result}: {row[result]}"
                    for result in headers["results"]
                ),
                end="\n\n",
            )
            part.results.append(
                Results(
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
                    version="after",
                )
            )

        parts.append(part)

    if parts[0].part_id == 0:
        print("Part IDs are missing in the data.")
        parts = add_part_ids(parts)

    if silver_reasoning_path:
        print("Adding silver reasoning to the data...")
        for part in parts:
            part.silver_reasoning = silver_reasoning.get(
                task_id=part.task_id, sample_id=part.sample_id, part_id=part.part_id
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
    # TODO: specify the path to silver reasoning if you wish to addition it
    silver_reasoning_path = "data/silver_reasoning/silver_reasoning_test.csv"
    run(
        data_path=data_path,
    )
