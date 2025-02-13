from typing import List

import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer

from data.DataLoader import DataLoader
from data.utils import *


class Statistics:
    def __init__(self):
        self.loader = DataLoader()
        self.lemmatize = WordNetLemmatizer().lemmatize

    def get_data_stats(self, data_path: str):
        """
        Get statistics of the data.

        :param data_path: path to the data
        :return:
        """
        data = self.loader.load_task_data(path=data_path, split="train")
        self.analyse_task_questions(data)

    @staticmethod
    def analyse_task_questions(data: dict):
        """
        Plot the number of questions for each task.

        :param data: dictionary containing the data
        """
        check_or_create_directory("../plots")
        q_stats = {}
        c_stats = {}
        c_before_q = {}
        for task in data.keys():
            q_stats[task] = []
            c_stats[task] = []
            c_before_q[task] = []
            for sample in data[task].keys():
                # given that each part contains one question, number of parts = number of questions
                q_stats[task].append(len(data[task][sample]))
                num_of_contexts = [len(part["context"]) for part in data[task][sample]]
                c_stats[task].append(sum(num_of_contexts))
                for part in sample:
                    for ix, line_num in enumerate(list(part["question"].keys())):
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

    def are_identical(self, true, prediction) -> bool:
        """
        Check if the prediction is correct in a rather strict way:
        - the answers are the identical in lower case OR
        - the answers are the identical after lemmatization OR
        - the answers are the same set of words OR
        - the answers are the same set of words after lemmatization

        :param true: true answer
        :param prediction: predicted answer
        :return: True if the prediction is correct, False otherwise
        """
        true, prediction = normalize_token(true), normalize_token(prediction)

        true_list = answer_into_list(true)
        prediction_list = answer_into_list(prediction)

        true_set, prediction_set = set(true_list), set(prediction_list)
        lemma_true_set = set(self.lemmatize(t) for t in true_set)
        lemma_pred_set = set(self.lemmatize(t) for t in prediction_set)

        if true == prediction:
            return True

        if two_true_one_pred(true_list, prediction_list):
            # this is for cases of two identical answers, like "east, east"
            # if the model outputs "east", it is not correct
            return False

        if true_set == prediction_set or lemma_true_set == lemma_pred_set:
            # this is needed for cases where the order of answer doesn't matter
            # e.g. the routes "east, south" and "south, east" should be both
            # considered correct because they lead to the same destination
            return True

        if true == normalize_numbers(prediction):
            return True

        return False

    @staticmethod
    def accuracy_score_by_bools(bool_predicted) -> float:
        """
        Compute the accuracy score by the proportion of True values.

        :param bool_predicted: boolean value
        :return: accuracy score
        """
        true_in_predicted = bool_predicted.count(True)
        return true_in_predicted / len(bool_predicted) if true_in_predicted else 0.0

    def accuracy_score(
        self, true_values: list[str], predicted_values: list[str]
    ) -> float:
        """
        Compute the accuracy score that also considers the order of the answers and lemmatization.

        :param true_values: list of true values
        :param predicted_values: list of predicted values
        :return: accuracy score
        """
        if len(true_values) == 0 or len(predicted_values) == 0:
            return 0.0

        true_in_predicted = 0
        for true, prediction in zip(true_values, predicted_values):
            if self.are_identical(true, prediction):
                true_in_predicted += 1

        return true_in_predicted / len(true_values) if true_in_predicted else 0.0

    def soft_match_accuracy_score(
        self, true_values: List[str], predicted_values: List[str]
    ) -> float:
        """
        Compute the accuracy score that also considers the order of the answers,
        and partial or wordy answers.

        :param true_values: list of true values
        :param predicted_values: list of predicted values
        :return: soft-match accuracy score
        """
        if len(true_values) == 0 or len(predicted_values) == 0:
            return 0.0

        true_in_predicted = 0
        for true, prediction in zip(true_values, predicted_values):
            true = normalize_token(true)
            prediction = normalize_token(prediction)
            true_set = set(answer_into_list(true))
            prediction_set = set(answer_into_list(prediction))

            if misleading_no(true_set, prediction_set):
                continue

            if self.are_identical(true, prediction):
                true_in_predicted += 1

            elif true in prediction:
                true_in_predicted += 0.75

            # for partial answer of questions with two supporting facts
            elif prediction in true or prediction_set.intersection(true_set):
                true_in_predicted += 0.5

        return true_in_predicted / len(true_values) if true_in_predicted else 0.0


if __name__ == "__main__":
    stats = Statistics()
    # stats.get_data_stats(data_path="tasks_1-20_v1-2/en")
    print(stats.are_identical("east, north", "east, then north"))
    print("Accuracy score:", stats.accuracy_score(["south, south"], ["south"]))
    print(
        "Sort-math accuracy score",
        stats.soft_match_accuracy_score(["south, south"], ["south"]),
    )
    print(stats.soft_match_accuracy_score(["one"], ["not mentioned"]))
