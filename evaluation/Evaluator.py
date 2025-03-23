from __future__ import annotations

from evaluation.Metrics import Accuracy, Metric
from evaluation.Statistics import Statistics


class Evaluator:
    """
    This class handles everything related to model's evaluation, like accuracy scores and metrics.
    """

    def __init__(self, level: str, after: bool = True) -> None:
        """
        Initialize the Evaluator.

        :param level: the level of evaluation
        :param after: whether the evaluation is done after the setting was applied to the model's output
        """
        self.level: str = level
        self.after: bool = after
        self.add_on = "after" if self.after else "before"
        self.stats: Statistics = Statistics()

        self.exact_match_accuracy: Accuracy = Accuracy(
            f"[{self.add_on}] Exact-Match Accuracy"
        )
        self.soft_match_accuracy: Accuracy = Accuracy(
            f"[{self.add_on}] Soft-Match Accuracy"
        )

        self.exact_match_std: Metric = Metric(
            f"Standard Deviation for Exact-Match Accuracy {self.add_on.capitalize()}"
        )
        self.soft_match_std: Metric = Metric(
            f"Standard Deviation for Soft-Match Accuracy {self.add_on.capitalize()}"
        )

    def __repr__(self):
        return (
            f"<{self.level.capitalize()} Evaluator [{self.add_on}]: "
            f"exact_match={self.exact_match_accuracy.get_mean()}, "
            f"soft_match={self.soft_match_accuracy.get_mean()}), "
            f"exact_match_std={self.exact_match_accuracy.get_mean()}, "
            f"soft_match_std={self.soft_match_accuracy.get_mean()}>"
        )

    def print_accuracies(self, id_, exact_match_acc=None, soft_match_acc=None) -> None:
        """
        Print the accuracy scores for the level and id.

        :return: None
        """
        if exact_match_acc and soft_match_acc:
            print(
                f"\n[{self.add_on}] Exact-match accuracy score for {self.level} {id_}:",
                round(exact_match_acc, 2),
            )
            print(
                f"[{self.add_on}] Soft-match accuracy score for {self.level} {id_}:",
                round(soft_match_acc, 2),
                end="\n\n",
            )
        else:
            print(
                f"\n[{self.add_on}] Exact-match accuracy score for {self.level} {id_}:",
                self.exact_match_accuracy.get_mean(),
                (
                    f"-- std: {self.exact_match_accuracy.get_std()}"
                    if len(self.exact_match_accuracy) > 1
                    else ""
                ),
            )
            print(
                f"[{self.add_on}] Soft-match accuracy score for {self.level} {id_}:",
                self.soft_match_accuracy.get_mean(),
                (
                    f"-- std: {self.soft_match_accuracy.get_std()}"
                    if len(self.soft_match_accuracy) > 1
                    else ""
                ),
                end="\n\n",
            )


class MetricEvaluator(Evaluator):
    """
    This class handles everything related to evaluation.
    """

    def __init__(self, level: str, after: bool = True) -> None:
        """
        Initialize the Evaluator.

        :param level: the level of evaluation
        """
        super().__init__(level, after)

    def update(self, smaller_evaluator: Evaluator) -> None:
        """
        Update the evaluator with the information from a lower-level evaluator.

        :param smaller_evaluator: the lower-level evaluator, e.g. a sample
        """
        self.exact_match_accuracy.add(smaller_evaluator.exact_match_accuracy)
        self.soft_match_accuracy.add(smaller_evaluator.soft_match_accuracy)
        self.exact_match_std.add(smaller_evaluator.exact_match_std)
        self.soft_match_std.add(smaller_evaluator.soft_match_std)


class AnswerEvaluator(Evaluator):
    """
    This class handles everything related to evaluation on the sample level.
    """

    def __init__(self, level: str, after: bool = True) -> None:
        """
        Initialize the SampleEvaluator.

        :param level: the level of evaluation
        :param after: whether the evaluation is done after the setting was applied to the model's output
        """
        super().__init__(level, after)

        self.pred_answers: list[str] = []
        self.pred_reasonings: list[str] = []

        self.golden_answers: list[str] = []
        self.silver_reasonings: list[str] = []

    def calculate_accuracies(self) -> tuple[float, float]:
        """
        Calculate the accuracy scores for the sample and print them.
        Used for levels sample and task.

        :return: tuple of exact-match and soft-match accuracy scores
        """
        exact_match_acc = self.stats.exact_match_accuracy_score(
            self.golden_answers, self.pred_answers
        )
        self.exact_match_accuracy.add(exact_match_acc)
        soft_match_acc = self.stats.soft_match_accuracy_score(
            self.golden_answers, self.pred_answers
        )
        self.soft_match_accuracy.add(soft_match_acc)
        return exact_match_acc, soft_match_acc
