from __future__ import annotations

from evaluation.Metrics import Accuracy, AttnDistribution, Metric, AttnOnTarget
from evaluation.Metrics import Accuracy, BLEU, Meteor, Metric, ROUGE
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

        self.max_supp_attn: AttnDistribution = AttnDistribution(
            f"Max Attention Distribution {self.version.capitalize()}"
        )
        self.max_supp_attn_std: Metric = Metric(
            f"Standard Deviation for Max Attention Distribution {self.version.capitalize()}"
        )

        self.attn_on_target: AttnOnTarget = AttnOnTarget(
            f"Attention on Target Tokens {self.version.capitalize()}"
        )
        self.attn_on_target_std: Metric = Metric(
            f"Standard Deviation for Attention on Target Tokens {self.version.capitalize()}"
        )

        self.bleu = BLEU(f"BLEU {self.add_on.capitalize()}")
        self.rouge = ROUGE(f"ROUGE {self.add_on.capitalize()}")
        self.meteor = Meteor(f"METEOR {self.add_on.capitalize()}")

        self.bleu_std = Metric(
            f"Standard Deviation for BLEU {self.add_on.capitalize()}"
        )
        self.rouge_std = Metric(
            f"Standard Deviation for ROUGE {self.add_on.capitalize()}"
        )
        self.meteor_std = Metric(
            f"Standard Deviation for METEOR {self.add_on.capitalize()}"
        )

    def __repr__(self):
        return (
            f"<{self.level.capitalize()} Evaluator [{self.version}]: "
            f"exact_match={self.exact_match_accuracy.get_mean()}, "
            f"soft_match={self.soft_match_accuracy.get_mean()}), "
            f"exact_match_std={self.exact_match_accuracy.get_std()}, "
            f"soft_match_std={self.soft_match_accuracy.get_std()}>, "
            f"max_supp_attn={self.max_supp_attn.get_mean()}>, "
            f"max_supp_attn_std={self.max_supp_attn.get_std()}>"
            f"attn_on_target={self.attn_on_target.get_mean()}>, "
            f"attn_on_target_std={self.attn_on_target.get_std()}>"
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
        self.max_supp_attn.add(smaller_evaluator.max_supp_attn)
        self.max_supp_attn_std.add(smaller_evaluator.max_supp_attn.get_std())
        self.attn_on_target.add(smaller_evaluator.attn_on_target)
        self.attn_on_target_std.add(smaller_evaluator.attn_on_target.get_std())

        self.bleu.add(smaller_evaluator.bleu)
        self.rouge.add(smaller_evaluator.rouge)
        self.meteor.add(smaller_evaluator.meteor)
        self.bleu_std.add(smaller_evaluator.bleu.get_std())
        self.rouge_std.add(smaller_evaluator.rouge.get_std())
        self.meteor_std.add(smaller_evaluator.meteor.get_std())

    def calculate_std(self):
        """
        Calculate the standard deviations for the metric.
        """
        exact_match_std = self.exact_match_accuracy.get_std()
        self.exact_match_std.add(exact_match_std)
        soft_match_std = self.soft_match_accuracy.get_std()
        self.soft_match_std.add(soft_match_std)
        max_supp_attn_std = self.max_supp_attn.get_std()
        self.max_supp_attn_std.add(max_supp_attn_std)
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
                f"bleu_{self.version}": self.bleu,
                f"rouge_{self.version}": self.rouge,
                f"bleu_std_{self.version}": self.bleu_std,
                f"rouge_std_{self.version}": self.rouge_std,
            }
        return {
            f"exact_match_accuracy_{self.version}": self.exact_match_accuracy.get_mean(),
            f"soft_match_accuracy_{self.version}": self.soft_match_accuracy.get_mean(),
            f"exact_match_std_{self.version}": self.exact_match_accuracy.get_std(),
            f"soft_match_std_{self.version}": self.soft_match_accuracy.get_std(),
            f"bleu_{self.version}": self.bleu.get_mean(),
            f"rouge_{self.version}": self.rouge.get_mean(),
            f"bleu_std_{self.version}": self.bleu.get_std(),
            f"rouge_std_{self.version}": self.rouge.get_std(),
        }

    def get_attentions(self, as_lists: bool = False) -> dict[str, float | Metric]:
        """
        Get the attention metrics for the evaluator.

        :return: the max attention distribution and attention on target
        """
        if as_lists:
            return {
                f"max_attn_dist_{self.version}": self.max_supp_attn,
                f"max_attn_dist_std_{self.version}": self.max_supp_attn_std,
                f"max_attn_on_target_{self.version}": self.attn_on_target,
                f"attn_on_target_std_{self.version}": self.attn_on_target_std,
            }
        return {
            f"max_supp_attn_{self.version}": self.max_supp_attn.get_mean(),
            f"max_supp_attn_std_{self.version}": self.max_supp_attn.get_std(),
            f"attn_on_target_{self.version}": self.attn_on_target.get_mean(),
            f"attn_on_target_std_{self.version}": self.attn_on_target_std.get_std(),
            f"exact_match_accuracy_{self.version}": self.exact_match_accuracy.get_mean(),
            f"soft_match_accuracy_{self.version}": self.soft_match_accuracy.get_mean(),
            f"exact_match_std_{self.version}": self.exact_match_accuracy.get_std(),
            f"soft_match_std_{self.version}": self.soft_match_accuracy.get_std(),
            f"bleu_{self.version}": self.bleu.get_mean(),
            f"rouge_{self.version}": self.rouge.get_mean(),
            f"bleu_std_{self.version}": self.bleu.get_std(),
            f"rouge_std_{self.version}": self.rouge.get_std(),
        }

    def get_scores(self, as_lists: bool = False) -> dict[str, float | Metric]:
        """
        Get the scores for the evaluator.
        :param as_lists:
        :return:
        """
        add_on = "after" if self.after else "before"
        if as_lists:
            return {
                f"bleu_{self.version}": self.bleu,
                f"rouge_{self.version}": self.rouge,
                f"meteor_{self.version}": self.meteor,
                f"bleu_std_{self.version}": self.bleu_std,
                f"rouge_std_{self.version}": self.rouge_std,
                f"meteor_{self.version}": self.meteor_std,
            }
        return {
            f"bleu_{self.version}": self.bleu.get_mean(),
            f"rouge_{self.version}": self.rouge.get_mean(),
            f"meteor_{self.version}": self.meteor.get_mean(),
            f"bleu_std_{self.version}": self.bleu.get_std(),
            f"rouge_std_{self.version}": self.rouge.get_std(),
            f"meteor_std_{self.version}": self.meteor.get_std(),
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

    def calculate_bleu(self) -> float:
        """
        Calculate the BLEU score for the sample part.

        :return: the BLEU score
        """
        bleu_score = self.bleu.bleu.compute(
            references=self.silver_reasonings, predictions=self.pred_reasonings
        )
        self.bleu.add(bleu_score)
        return bleu_score

    def calculate_rouge(self) -> float:
        """
        Calculate the ROUGE score for the sample part.

        :return: the ROUGE score
        """
        rouge_score = self.rouge.rouge.compute(
            references=self.silver_reasonings, predictions=self.pred_reasonings
        )
        self.rouge.add(rouge_score)
        return rouge_score

    def calculate_meteor(self):
        """
        Calculate the Meteor score for the sample part.

        :return: the Meteor score
        """
        meteor_score = self.meteor.meteor.compute(
            references=self.silver_reasonings, predictions=self.pred_reasonings
        )
        self.rouge.add(meteor_score)
        return meteor_score
