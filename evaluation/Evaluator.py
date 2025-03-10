import re

import en_core_web_sm

nlp = en_core_web_sm.load()

from evaluation.Accuracy import Accuracy
from evaluation.Statistics import Statistics


class Evaluator:
    """
    This class handles everything related to model's evaluation, like accuracy scores and metrics.
    """

    def __init__(self, level: str) -> None:
        """
        Initialize the Evaluator.

        :param level: the level of evaluation
        """
        self.level = level
        self.stats = Statistics()

        self.exact_match_accuracy = Accuracy("exact_match")
        self.soft_match_accuracy = Accuracy("soft_match")

        self.there = 0
        self.verbs = 0
        self.pronouns = 0
        self.not_mentioned = 0

    def print_accuracies(self, id_, exact_match_acc=None, soft_match_acc=None) -> None:
        """
        Print the accuracy scores for the level and id.

        :return: None
        """
        id_ = id_ + 1 if type(id_) == int else id_
        if exact_match_acc and soft_match_acc:
            print(
                f"\nExact-match accuracy score for {self.level} {id_}:",
                round(exact_match_acc, 2),
            )
            print(
                f"Soft-match accuracy score for {self.level} {id_}:",
                round(soft_match_acc, 2),
                end="\n\n",
            )
        else:
            print(
                f"\nExact-match accuracy score for {self.level} {id_}:",
                self.exact_match_accuracy.get_mean(),
                (
                    f"-- std: {self.exact_match_accuracy.get_std()}"
                    if len(self.exact_match_accuracy) > 1
                    else ""
                ),
            )
            print(
                f"Soft-match accuracy score for {self.level} {id_}:",
                self.soft_match_accuracy.get_mean(),
                (
                    f"-- std: {self.soft_match_accuracy.get_std()}"
                    if len(self.soft_match_accuracy) > 1
                    else ""
                ),
                end="\n\n",
            )

    def print_metrics(self, id_) -> None:
        """
        Print the metrics for the level and id.

        :return: None
        """
        self.print_accuracies(id_)
        print("Number of 'there':", self.there)
        print("Number of 'verbs':", self.verbs)
        print("Number of 'pronouns':", self.pronouns)
        print("Number of 'not mentioned':", self.not_mentioned)
        print("______________", end="\n\n")


class MetricEvaluator(Evaluator):
    """
    This class handles everything related to evaluation.
    """

    def __init__(self, level: str) -> None:
        """
        Initialize the Evaluator.

        :param level: the level of evaluation
        """
        super().__init__(level)

    def update(self, smaller_evaluator: Evaluator) -> None:
        """
        Update the evaluator with the information from a lower-level evaluator.

        :param smaller_evaluator: the lower-level evaluator, e.g. a sample
        """
        self.exact_match_accuracy.add(smaller_evaluator.exact_match_accuracy)
        self.soft_match_accuracy.add(smaller_evaluator.soft_match_accuracy)

        self.there += smaller_evaluator.there
        self.verbs += smaller_evaluator.verbs
        self.pronouns += smaller_evaluator.pronouns
        self.not_mentioned += smaller_evaluator.not_mentioned


class AnswerEvaluator(Evaluator):
    """
    This class handles everything related to evaluation on the sample level.
    """

    def __init__(self, level: str) -> None:
        """
        Initialize the SampleEvaluator.

        :param level: the level of evaluation
        """
        super().__init__(level)

        self.true_values = []
        self.predicted_values = []
        self.reasonings = []

    def evaluate(self, true, answer):
        """
        Evaluate the answer by checking
        - if it is correct,
        - if 'there' is mentioned,
        - if a verb is used instead of a noun,
        - if 'not mentioned' is mentioned,
        - if pronouns are used (instead of names).

        :param true: the true answer
        :param answer: the predicted answer
        :return: dictionary with evaluation results
        """
        return {
            "correct?": self.stats.are_identical(true, answer),
            "there": self.if_there(answer),
            "verbs": self.if_verb(answer),
            "pronouns": self.if_pronouns(answer),
            "not_mentioned": self.if_not_mentioned(answer),
        }

    def if_there(self, answer) -> bool:
        """
        Check if 'there', 'here' or 'nowhere' is in an answer.

        :param answer: the answer
        :return: bool
        """
        if answer and re.search(r"\b((?:now|t)?here)\b", answer):
            self.there += 1
            return True
        return False

    def if_verb(self, answer) -> bool:
        """
        Check if a verb is in the answer.
        """
        answer = nlp(answer)
        if answer and answer[0].tag_.startswith("VB"):
            self.verbs += 1
            return True
        return False

    def if_pronouns(self, answer) -> bool:
        """
        Check if 'he', 'she', 'it', 'they' or 'we' appears in the answer.

        :param answer: the answer
        :return: bool
        """
        if answer and re.search(r"\b(?:he|she|it|her|him|they|them)\b", answer):
            self.pronouns += 1
            return True
        return False

    def if_not_mentioned(self, answer) -> bool:
        """
        Check if the model states in doesn't know the answer through phrases like
         - 'not mentioned'
         - 'no information'
         - 'unknown'

        :param answer: the answer
        :return: bool
        """
        if answer and re.search(r"\bmention|information|unknown", answer):
            self.not_mentioned += 1
            return True
        return False

    def calculate_accuracies(self) -> tuple[float, float]:
        """
        Calculate the accuracy scores for the sample and print them.
        Used for levels sample and task

        :return: tuple of exact-match and soft-match accuracy scores
        """
        exact_match_acc = self.stats.exact_match_accuracy_score(
            self.true_values, self.predicted_values
        )
        self.exact_match_accuracy.add(exact_match_acc)
        soft_match_acc = self.stats.soft_match_accuracy_score(
            self.true_values, self.predicted_values
        )
        self.soft_match_accuracy.add(soft_match_acc)
        return exact_match_acc, soft_match_acc
