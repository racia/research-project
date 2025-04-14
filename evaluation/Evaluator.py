from __future__ import annotations

from evaluation.Metrics import Accuracy, Metric, AttnDistribution
from evaluation.Statistics import Statistics


class Evaluator:
    """
    This class handles everything related to model's evaluation, like accuracy scores and metrics.
    """

    def __init__(self, level: str, version: str = "after") -> None:
        """
        Initialize the Evaluator.

        :param level: the level of evaluation
        :param version: "before" or "after" the setting was applied
        """
        self.level: str = level
        if version not in ["before", "after"]:
            raise ValueError(f"Version must be 'before' or 'after', not {version}.")
        self.version: str = version
        self.stats: Statistics = Statistics()

        self.exact_match_accuracy: Accuracy = Accuracy(
            f"[{self.version}] Exact-Match Accuracy"
        )
        self.soft_match_accuracy: Accuracy = Accuracy(
            f"[{self.version}] Soft-Match Accuracy"
        )

        self.exact_match_std: Metric = Metric(
            f"Standard Deviation for Exact-Match Accuracy {self.version.capitalize()}"
        )
        self.soft_match_std: Metric = Metric(
            f"Standard Deviation for Soft-Match Accuracy {self.version.capitalize()}"
        )

        self.max_attn_dist: AttnDistribution = AttnDistribution(
            f"Max Attention Distribution {self.version.capitalize()}"
        )
        self.max_attn_dist_std: Metric = Metric(
            f"Standard Deviation for Max Attention Distribution {self.version.capitalize()}"
        )

    def __repr__(self):
        return (
            f"<{self.level.capitalize()} Evaluator [{self.version}]: "
            f"exact_match={self.exact_match_accuracy.get_mean()}, "
            f"soft_match={self.soft_match_accuracy.get_mean()}), "
            f"exact_match_std={self.exact_match_accuracy.get_std()}, "
            f"soft_match_std={self.soft_match_accuracy.get_std()}>, "
            f"max_attn_dist={self.max_attn_dist.get_mean()}>, "
            f"max_attn_dist_std={self.max_attn_dist.get_std()}>"
        )

    def print_accuracies(self, id_, exact_match_acc=None, soft_match_acc=None) -> None:
        """
        Print the accuracy scores for the level and id.

        :return: None
        """
        if exact_match_acc and soft_match_acc:
            print(
                f"\n[{self.version}] Exact-match accuracy score for {self.level} {id_}:",
                round(exact_match_acc, 2),
            )
            print(
                f"[{self.version}] Soft-match accuracy score for {self.level} {id_}:",
                round(soft_match_acc, 2),
            )
        else:
            print(
                f"\n[{self.version}] Exact-match accuracy score for {self.level} {id_}:",
                self.exact_match_accuracy.get_mean(),
                (
                    f"-- std: {self.exact_match_accuracy.get_std()}"
                    if len(self.exact_match_accuracy) > 1
                    else ""
                ),
            )
            print(
                f"[{self.version}] Soft-match accuracy score for {self.level} {id_}:",
                self.soft_match_accuracy.get_mean(),
                (
                    f"-- std: {self.soft_match_accuracy.get_std()}"
                    if len(self.soft_match_accuracy) > 1
                    else ""
                ),
            )
        if self.max_attn_dist:
            print(
                f"[{self.version}] Max attention distribution for {self.level} {id_}:",
                self.max_attn_dist.get_mean(),
                (
                    f"-- std: {self.max_attn_dist.get_std()}"
                    if len(self.max_attn_dist) > 1
                    else ""
                ),
                end="\n\n",
            )


class MetricEvaluator(Evaluator):
    """
    This class handles everything related to evaluation.
    """

    def __init__(self, level: str, version: str = "after") -> None:
        """
        Initialize the Evaluator.

        :param level: the level of evaluation
        :param version: "before" or "after" the setting was applied
        """
        super().__init__(level, version)

    def update(self, smaller_evaluator: Evaluator) -> None:
        """
        Update the evaluator with the information from a lower-level evaluator.

        :param smaller_evaluator: the lower-level evaluator, e.g. a sample
        """
        self.exact_match_accuracy.add(smaller_evaluator.exact_match_accuracy)
        self.soft_match_accuracy.add(smaller_evaluator.soft_match_accuracy)
        self.exact_match_std.add(smaller_evaluator.exact_match_accuracy.get_std())
        self.soft_match_std.add(smaller_evaluator.soft_match_accuracy.get_std())
        self.max_attn_dist.add(smaller_evaluator.max_attn_dist)
        self.max_attn_dist_std.add(smaller_evaluator.max_attn_dist.get_std())

    def calculate_std(self):
        """
        Calculate the standard deviations for the metric.
        """
        exact_match_std = self.exact_match_accuracy.get_std()
        self.exact_match_std.add(exact_match_std)
        soft_match_std = self.soft_match_accuracy.get_std()
        self.soft_match_std.add(soft_match_std)
        max_attn_dist_std = self.max_attn_dist.get_std()
        self.max_attn_dist_std.add(max_attn_dist_std)
        return exact_match_std, soft_match_std

    def get_accuracies(self, as_lists: bool = False) -> dict[str, float | Metric]:
        """
        Get the metrics for the evaluator.

        :return: the exact-match and soft-match accuracy scores
        """
        if as_lists:
            return {
                f"exact_match_accuracy_{self.version}": self.exact_match_accuracy,
                f"soft_match_accuracy_{self.version}": self.soft_match_accuracy,
                f"exact_match_std_{self.version}": self.exact_match_std,
                f"soft_match_std_{self.version}": self.soft_match_std,
            }
        return {
            f"exact_match_accuracy_{self.version}": self.exact_match_accuracy.get_mean(),
            f"soft_match_accuracy_{self.version}": self.soft_match_accuracy.get_mean(),
            f"exact_match_std_{self.version}": self.exact_match_accuracy.get_std(),
            f"soft_match_std_{self.version}": self.soft_match_accuracy.get_std(),
        }

    def get_max_attn_dist(self, as_lists: bool = False) -> dict[str, float | Metric]:
        """
        Get the max attention distribution for the evaluator.

        :return: the max attention distribution
        """
        if as_lists:
            return {
                f"max_attn_dist_{self.version}": self.max_attn_dist,
                f"max_attn_dist_std_{self.version}": self.max_attn_dist_std,
            }
        return {
            f"max_attn_dist_{self.version}": self.max_attn_dist.get_mean(),
            f"max_attn_dist_std_{self.version}": self.max_attn_dist.get_std(),
        }


class AnswerEvaluator(Evaluator):
    """
    This class handles everything related to evaluation on the sample level.
    """

    def __init__(self, level: str, version: str = "after") -> None:
        """
        Initialize the SampleEvaluator.

        :param level: the level of evaluation
        :param version: "before" or "after" the setting was applied
        """
        super().__init__(level, version)

        self.pred_answers: list[str] = []
        self.pred_reasonings: list[str] = []

        self.golden_answers: list[str] = []
        self.silver_reasonings: list[str] = []

    def calculate_accuracies(self) -> tuple[float, ...]:
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
