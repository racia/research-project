# Description: This script is used to analyse the number of questions per task.
# The script will also plot for each task:
# - the total number of questions;
# - the total number of lines of context for each task;
# - the number of lines of context before each question.
from pathlib import Path

from pathlib import Path

import matplotlib.pyplot as plt

from data.DataLoader import DataLoader
from evaluation.utils import check_or_create_directory

PREFIX = Path.cwd()
while PREFIX.name != "research-project":
    PREFIX = PREFIX.parent


def run(data_path: str):
    """
    Plot the number of questions for each task.

    :param data_path: path to the data
    :return: None
    """
    loader = DataLoader()
    path = f"{PREFIX}/data/{data_path}"
    data = loader.load_task_data(path=path, split="test")
    check_or_create_directory(f"{PREFIX}/plots")
    q_stats = {}
    c_stats = {}
    c_before_q = {}
    amount_of_samples = {}
    questions_per_sample = {}
    for task in data.keys():
        amount_of_samples[task] = 0
        questions_per_sample[task] = []
        q_stats[task] = []
        c_stats[task] = []
        c_before_q[task] = []
        for sample_id in data[task].keys():
            questions_per_sample[task].append(0)
            sample = data[task][sample_id]
            # given that each part contains one question, number of parts = number of questions
            q_stats[task].append(len(data[task][sample_id]))
            num_of_contexts = [len(part["context"]) for part in data[task][sample_id]]
            c_stats[task].append(sum(num_of_contexts))
            for part in sample:
                for ix, line_num in enumerate(list(part["question"].keys())):
                    questions_per_sample[task][-1] += 1
                    if ix == 0:
                        c_before_q[task].append(line_num - 1)  # first question
                    else:
                        c_before_q[task].append(
                            line_num - ix
                        )  # subtract amount of questions before this one
            amount_of_samples[task] += 1

    plt.figure(figsize=(10, 5))
    plt.bar(q_stats.keys(), [sum(q_stats[task]) for task in q_stats.keys()])
    plt.xlabel("Task")
    plt.ylabel("Number of Questions")
    plt.title("Total Number of Questions per Task")
    plt.savefig("plots/num_questions_per_task.png")

    plt.figure(figsize=(10, 5))
    plt.boxplot(q_stats.values())
    plt.xticks(range(1, len(q_stats.keys()) + 1), q_stats.keys())
    plt.xlabel("Task")
    plt.ylabel("Amount of Questions")
    plt.title("Amount of Questions per Task")
    plt.savefig("plots/questions_per_task.png")

    plt.figure(figsize=(10, 5))
    plt.boxplot(c_before_q.values())
    plt.xticks(range(1, len(q_stats.keys()) + 1), c_before_q.keys())
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
        tick_label=amount_of_samples.keys(),
    )
    plt.xlabel("Task")
    plt.ylabel("Amount of Samples")
    plt.title("Amount of Samples per Task")
    plt.savefig("plots/num_samples_per_task.png")
    print(amount_of_samples)

    plt.figure(figsize=(10, 5))
    plt.boxplot(questions_per_sample.values())
    plt.xticks(range(1, len(q_stats.keys()) + 1), questions_per_sample.keys())
    plt.xlabel("Task")
    plt.ylabel("Amount of Questions per Sample")
    plt.title("Amount of Questions per Sample")
    plt.savefig("plots/questions_per_sample.png")


if __name__ == "__main__":
    run(data_path="tasks_1-20_v1-2/en-valid")
