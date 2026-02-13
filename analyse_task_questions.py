from pathlib import Path

import matplotlib.pyplot as plt

from data.DataLoader import DataLoader
from evaluation.utils import check_or_create_directory
from settings.config import Enumerate

PREFIX = Path.cwd()
while PREFIX.name != "research-project":
    PREFIX = PREFIX.parent


def run(data_path: str):
    enum = Enumerate(context=True, question=False)
    loader = DataLoader(samples_per_task=100, to_enumerate=enum, prefix=PREFIX)
    path = f"{PREFIX}/data/{data_path}"
    data = loader.load_task_data(
        path=path,
        split="test",
        multi_system=False,
        tasks=list(range(1, 21)),
    )

    check_or_create_directory(f"{PREFIX}/plots")
    check_or_create_directory(f"{PREFIX}/evaluation/plots")

    q_stats = {}
    c_stats = {}
    c_before_q = {}
    amount_of_samples = {}
    questions_per_sample = {}

    min_context = float("inf")
    max_context = float("-inf")
    min_info = []
    max_info = []

    for task_id, samples_dict in data.items():
        amount_of_samples[task_id] = len(samples_dict)
        q_stats[task_id] = []
        c_stats[task_id] = []
        c_before_q[task_id] = []
        questions_per_sample[task_id] = []

        for sample_id, sample_parts in samples_dict.items():
            num_questions = len(sample_parts)
            q_stats[task_id].append(num_questions)
            questions_per_sample[task_id].append(num_questions)

            cumulative_context = []

            for q_index, sample_part in enumerate(sample_parts, start=1):
                cumulative_context.extend(sample_part.context_line_nums)
                num_context_before_q = len(cumulative_context)

                c_stats[task_id].append(len(sample_part.structured_context))
                c_before_q[task_id].append(num_context_before_q)

                if num_context_before_q < min_context:
                    min_context = num_context_before_q
                    min_info = [(task_id, sample_id, q_index)]
                elif num_context_before_q == min_context:
                    min_info.append((task_id, sample_id, q_index))

                if num_context_before_q > max_context:
                    max_context = num_context_before_q
                    max_info = [(task_id, sample_id, q_index)]
                elif num_context_before_q == max_context:
                    max_info.append((task_id, sample_id, q_index))

    plt.figure(figsize=(10, 5))
    plt.bar(q_stats.keys(), [sum(q_stats[task]) for task in q_stats.keys()])
    plt.xlabel("Task")
    plt.ylabel("Number of Questions")
    plt.title("Total Number of Questions per Task")
    plt.savefig("plots/num_questions_per_task.png")

    plt.figure(figsize=(10, 5))
    plt.boxplot(q_stats.values())
    plt.xticks(range(1, len(q_stats) + 1), list(q_stats.keys()))
    plt.xlabel("Task")
    plt.ylabel("Amount of Questions")
    plt.title("Amount of Questions per Task")
    plt.savefig("plots/questions_per_task.png")

    plt.figure(figsize=(10, 5))
    plt.boxplot(c_before_q.values())
    plt.xticks(range(1, len(c_before_q) + 1), list(c_before_q.keys()))
    plt.xlabel("Task")
    plt.ylabel("Lines of Context Before Question")
    plt.title("Lines of Context Before Question per Task")
    plt.savefig("plots/context_before_question_per_task.png")

    plt.figure(figsize=(10, 5))
    plt.bar(c_stats.keys(), [sum(c_stats[task]) for task in c_stats.keys()])
    plt.xlabel("Task")
    plt.ylabel("Total Lines of Context")
    plt.title("Total Lines of Context per Task")
    plt.savefig("plots/num_context_per_task.png")

    plt.figure(figsize=(10, 5))
    plt.bar(
        amount_of_samples.keys(),
        amount_of_samples.values(),
        tick_label=list(amount_of_samples.keys()),
    )
    plt.xlabel("Task")
    plt.ylabel("Amount of Samples")
    plt.title("Amount of Samples per Task")
    plt.savefig("plots/num_samples_per_task.png")
    print(amount_of_samples)

    plt.figure(figsize=(10, 5))
    plt.boxplot(questions_per_sample.values())
    plt.xticks(
        range(1, len(questions_per_sample) + 1), list(questions_per_sample.keys())
    )
    plt.xlabel("Task")
    plt.ylabel("Amount of Questions per Sample")
    plt.title("Amount of Questions per Sample")
    plt.savefig("plots/questions_per_sample.png")

    write_stats_to_txt(
        output_path=f"{PREFIX}/evaluation/plots/task_statistics.txt",
        q_stats=q_stats,
        c_stats=c_stats,
        c_before_q=c_before_q,
        amount_of_samples=amount_of_samples,
        questions_per_sample=questions_per_sample,
        min_context=min_context,
        min_info=min_info,
        max_context=max_context,
        max_info=max_info,
    )


def write_stats_to_txt(
    output_path: str,
    q_stats: dict,
    c_stats: dict,
    c_before_q: dict,
    amount_of_samples: dict,
    questions_per_sample: dict,
    min_context: int,
    min_info: list,
    max_context: int,
    max_info: list,
):
    with open(output_path, "w") as f:
        for task in q_stats.keys():
            f.write(f"Task: {task}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Number of samples: {amount_of_samples[task]}\n")
            f.write(f"Total number of questions: {sum(q_stats[task])}\n")
            f.write(f"Questions per sample: {q_stats[task]}\n")
            f.write(f"Questions per sample (counted): {questions_per_sample[task]}\n")
            f.write(f"Total lines of context: {sum(c_stats[task])}\n")
            f.write(f"Lines of context per sample: {c_stats[task]}\n")
            f.write(f"Lines of context before each question: {c_before_q[task]}\n\n\n")

        f.write(
            f"Lines of context before question: min: {min_context} in "
            + ", ".join([f"task {t}, sample {s}, question {q}" for t, s, q in min_info])
            + "\n"
            f"max: {max_context} in "
            + ", ".join([f"task {t}, sample {s}, question {q}" for t, s, q in max_info])
            + ".\n"
        )


if __name__ == "__main__":
    run(data_path="tasks_1-20_v1-2/en-valid")
