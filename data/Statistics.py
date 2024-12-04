import os

import matplotlib.pyplot as plt

from data.DataHandler import DataHandler


class Statistics:
    def __init__(self):
        self.dh = DataHandler()

    def check_or_create_directory(self, path: str):
        """
        Check if the directory exists, if not create it.

        :param path: path to the directory
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def get_data_stats(self, data_path: str):
        """
        Get statistics of the data.

        :param data_path: path to the data
        :return:
        """
        all_data = self.dh.read_data(path=data_path, train=True)
        processed_data = self.dh.process_data(all_data)

        self.analyse_task_questions(processed_data)

    def analyse_task_questions(self, data: dict):
        """
        Plot the number of questions for each task.

        :param data: dictionary containing the data
        """
        self.check_or_create_directory("../plots")
        q_stats = {}
        c_stats = {}
        c_before_q = {}
        for task in data.keys():
            q_stats[task] = []
            c_stats[task] = []
            c_before_q[task] = []
            for id in data[task].keys():
                q_stats[task].append(len(data[task][id]["question"]))
                c_stats[task].append(len(data[task][id]["context"]))
                for ix, line_num in enumerate(list(data[task][id]["question"].keys())):
                    if ix == 0:
                        c_before_q[task].append(line_num - 1)  # first question
                    else:
                        c_before_q[task].append(
                            line_num - ix
                        )  # subtract amount of questions before this one

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


if __name__ == "__main__":
    stats = Statistics()
    stats.get_data_stats(data_path="tasks_1-20_v1-2/en")
