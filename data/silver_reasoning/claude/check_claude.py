import collections
import csv
import json
from pathlib import Path

from data.DataLoader import DataLoader
from evaluation.Statistics import Statistics

PREFIX = Path.cwd()
while PREFIX.name != "research-project":
    PREFIX = PREFIX.parent
print("Current path", PREFIX)


def read_claude_data() -> dict[tuple[int, int, int], dict]:
    data = dict()
    for i in range(1, 21):
        with open(
            f"{PREFIX}/data/silver_reasoning/claude/silver_reasoning_test_{i}.csv",
            "r",
        ) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row["task_id"] = int(row["task_id"])
                row["sample_id"] = int(row["sample_id"])
                row["part_id"] = int(row["part_id"])
                data[(row["task_id"], row["sample_id"], row["part_id"])] = row
                if "silver_reasoning" in row.keys():
                    row["reasoning"] = row.pop("silver_reasoning")
                if "task" in row.keys():
                    row.pop("task")

                if row["sample_id"] > 100:
                    break

    return data


stats = Statistics()
loader = DataLoader(samples_per_task=100, prefix=PREFIX)

raw_data = loader.load_task_data(
    path=f"{PREFIX}/../tasks_1-20_v1-2/en-valid/",
    split="test",
    tasks=[i for i in range(1, 21)],
    multi_system=False,
    lookup=True,
)
print("Raw data length", len(raw_data), type(raw_data))

claude_data = read_claude_data()

absent_keys = []
incorrect_answers = dict()
print("Length of Claude data:", len(claude_data))
task_counter = collections.Counter()
for key in raw_data.keys():
    if key not in claude_data.keys():
        absent_keys.append(key)
    else:
        golden_answer = raw_data[key].golden_answer
        if not stats.are_identical(golden_answer, claude_data[key]["answer"]):
            task_counter[claude_data[key]["task_id"]] += 1
            joined_key = ",".join(map(str, key))
            incorrect_answers[joined_key] = claude_data[key]
            incorrect_answers[joined_key]["golden_answer"] = golden_answer

print("Absent keys:", len(absent_keys))
print("Incorrect answers:", len(incorrect_answers))
print("Task counter:", task_counter)
print(json.dumps(incorrect_answers, indent=4))

with open(
    f"{PREFIX}/data/silver_reasoning/claude/incorrect_answers.csv", "w"
) as csvfile:
    fieldnames = [
        "task_id",
        "sample_id",
        "part_id",
        "reasoning",
        "answer",
        "golden_answer",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    for part in incorrect_answers.values():
        writer.writerow(part)

# with open(f"{PREFIX}/data/silver_reasoning/claude/absent_keys.csv", "w") as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=["task_id", "sample_id", "part_id"])
#     writer.writeheader()
#     for task_id, sample_id, part_id in absent_keys:
#         writer.writerow(
#             {"task_id": task_id, "sample_id": sample_id, "part_id": part_id}
#         )
