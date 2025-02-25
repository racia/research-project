from typing import List

from nltk.stem import WordNetLemmatizer

from evaluation.utils import *


class Statistics:
    """
    Class for computing the statistics of the model.
    """

    def __init__(self):
        """
        Initialize the Statistics class.
        """
        self.lemmatize = WordNetLemmatizer().lemmatize

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

    def exact_match_accuracy_score(
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
    print(stats.are_identical("east, north", "east, then north"))
    print(
        "Accuracy score:", stats.exact_match_accuracy_score(["south, south"], ["south"])
    )
    print(
        "Sort-math accuracy score",
        stats.soft_match_accuracy_score(["south, south"], ["south"]),
    )
    print(stats.soft_match_accuracy_score(["one"], ["not mentioned"]))
