from __future__ import annotations

from evaluation.Metrics import Accuracy, BLEU, Meteor, Metric, ROUGE
from evaluation.Metrics import AttnDistribution, AttnOnTarget
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
            f"[{self.version}] Exact-Match Accuracy", "exact_match_accuracy"
        )
        self.soft_match_accuracy: Accuracy = Accuracy(
            f"[{self.version}] Soft-Match Accuracy", "soft_match_accuracy"
        )

        self.exact_match_std: Metric = Metric(
            f"Standard Deviation for Exact-Match Accuracy {self.version.capitalize()}",
            "exact_match_std",
        )
        self.soft_match_std: Metric = Metric(
            f"Standard Deviation for Soft-Match Accuracy {self.version.capitalize()}",
            "soft_match_std",
        )

        self.max_supp_attn: AttnDistribution = AttnDistribution(
            f"Max Attention Distribution {self.version.capitalize()}", "max_supp_attn"
        )
        self.max_supp_attn_std: Metric = Metric(
            f"Standard Deviation for Max Attention Distribution {self.version.capitalize()}",
            "max_supp_attn_std",
        )

        self.attn_on_target: AttnOnTarget = AttnOnTarget(
            f"Attention on Target Tokens {self.version.capitalize()}", "attn_on_target"
        )
        self.attn_on_target_std: Metric = Metric(
            f"Standard Deviation for Attention on Target Tokens {self.version.capitalize()}",
            "attn_on_target_std",
        )

        self.bleu = BLEU(f"BLEU {self.version.capitalize()}", "bleu")
        self.bleu_std = Metric(
            f"Standard Deviation for BLEU {self.version.capitalize()}", "bleu_std"
        )
        self.rouge = ROUGE(f"ROUGE {self.version.capitalize()}", "rouge")
        self.rouge_std = Metric(
            f"Standard Deviation for ROUGE {self.version.capitalize()}", "rouge_std"
        )
        self.meteor = Meteor(f"METEOR {self.version.capitalize()}", "meteor")
        self.meteor_std = Metric(
            f"Standard Deviation for METEOR {self.version.capitalize()}", "meteor_std"
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
            f"attn_on_target_std={self.attn_on_target.get_std()}>, "
            f"bleu={self.bleu.get_mean()}>, "
            f"bleu_std={self.bleu.get_std()}>, "
            f"rouge={self.rouge.get_mean()}>, "
            f"rouge_std={self.rouge.get_std()}>, "
            f"meteor={self.meteor.get_mean()}>, "
            f"meteor_std={self.meteor.get_std()}>"
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
        self.exact_match_std.add(smaller_evaluator.exact_match_accuracy.get_std())
        self.soft_match_accuracy.add(smaller_evaluator.soft_match_accuracy)
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
        attn_on_target_std = self.attn_on_target.get_std()
        self.attn_on_target_std.add(attn_on_target_std)
        bleu_std = self.bleu.get_std()
        self.bleu_std.add(bleu_std)
        rouge_std = self.rouge.get_std()
        self.rouge_std.add(rouge_std)
        meteor_std = self.meteor.get_std()
        self.meteor_std.add(meteor_std)

    def get_metrics(self, as_lists: bool = False) -> dict[str, Accuracy | Metric]:
        """
        Get the metrics for the evaluator.

        :param as_lists: if True, return the scores as lists
        :return: the exact-match and soft-match accuracy scores
        """
        if as_lists:
            return {
                f"exact_match_accuracy_{self.version}": self.exact_match_accuracy,
                f"exact_match_std_{self.version}": self.exact_match_std,
                f"soft_match_accuracy_{self.version}": self.soft_match_accuracy,
                f"soft_match_std_{self.version}": self.soft_match_std,
                f"max_attn_dist_{self.version}": self.max_supp_attn,
                f"max_attn_dist_std_{self.version}": self.max_supp_attn_std,
                f"max_attn_on_target_{self.version}": self.attn_on_target,
                f"attn_on_target_std_{self.version}": self.attn_on_target_std,
                f"bleu_{self.version}": self.bleu,
                f"bleu_std_{self.version}": self.bleu_std,
                f"rouge_{self.version}": self.rouge,
                f"rouge_std_{self.version}": self.rouge_std,
                f"meteor_{self.version}": self.meteor,
                f"meteor_std_{self.version}": self.meteor_std,
            }
        return {
            f"exact_match_accuracy_{self.version}": self.exact_match_accuracy.get_mean(),
            f"exact_match_std_{self.version}": self.exact_match_accuracy.get_std(),
            f"soft_match_accuracy_{self.version}": self.soft_match_accuracy.get_mean(),
            f"soft_match_std_{self.version}": self.soft_match_accuracy.get_std(),
            f"max_attn_dist_{self.version}": self.max_supp_attn.get_mean(),
            f"max_attn_dist_std_{self.version}": self.max_supp_attn.get_std(),
            f"max_attn_on_target_{self.version}": self.attn_on_target.get_mean(),
            f"attn_on_target_std_{self.version}": self.attn_on_target.get_std(),
            f"bleu_{self.version}": self.bleu.get_mean(),
            f"bleu_std_{self.version}": self.bleu.get_std(),
            f"rouge_{self.version}": self.rouge.get_mean(),
            f"rouge_std_{self.version}": self.rouge.get_std(),
            f"meteor_{self.version}": self.meteor.get_mean(),
            f"meteor_std_{self.version}": self.meteor.get_std(),
        }

    def get_accuracies(self, as_lists: bool = False) -> dict[str, float | Metric]:
        """
        Get the accuracy metrics for the evaluator.

        :param as_lists: if True, return the scores as lists
        :return: the exact-match and soft-match accuracy scores
        """
        if as_lists:
            return {
                f"exact_match_accuracy_{self.version}": self.exact_match_accuracy,
                f"exact_match_std_{self.version}": self.exact_match_std,
                f"soft_match_accuracy_{self.version}": self.soft_match_accuracy,
                f"soft_match_std_{self.version}": self.soft_match_std,
            }
        return {
            f"exact_match_accuracy_{self.version}": self.exact_match_accuracy.get_mean(),
            f"exact_match_std_{self.version}": self.exact_match_accuracy.get_std(),
            f"soft_match_accuracy_{self.version}": self.soft_match_accuracy.get_mean(),
            f"soft_match_std_{self.version}": self.soft_match_accuracy.get_std(),
        }

    def get_attentions(self, as_lists: bool = False) -> dict[str, float | Metric]:
        """
        Get the attention metrics for the evaluator.

        :param as_lists: if True, return the scores as lists
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
        }

    def get_reasoning_scores(self, as_lists: bool = False) -> dict[str, float | Metric]:
        """
        Get the reasoning scores for the evaluator.
        :param as_lists: if True, return the scores as lists
        :return: the reasoning scores
        """
        if as_lists:
            return {
                f"bleu_{self.version}": self.bleu,
                f"bleu_std_{self.version}": self.bleu_std,
                f"rouge_{self.version}": self.rouge,
                f"rouge_std_{self.version}": self.rouge_std,
                f"meteor_{self.version}": self.meteor,
                f"meteor_std_{self.version}": self.meteor_std,
            }
        return {
            f"bleu_{self.version}": self.bleu.get_mean(),
            f"bleu_std_{self.version}": self.bleu.get_std(),
            f"rouge_{self.version}": self.rouge.get_mean(),
            f"rouge_std_{self.version}": self.rouge.get_std(),
            f"meteor_{self.version}": self.meteor.get_mean(),
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
        self.exact_match_std.add(self.exact_match_accuracy.get_std())

        soft_match_acc = self.stats.soft_match_accuracy_score(
            self.golden_answers, self.pred_answers
        )
        self.soft_match_accuracy.add(soft_match_acc)
        self.soft_match_std.add(self.soft_match_accuracy.get_std())

        return exact_match_acc, soft_match_acc

    def calculate_attention(self) -> tuple[float, ...]:
        """
        Calculate the attention scores for the sample and print them.
        Used for levels sample and task.

        :return: tuple of max attention distribution and attention on target
        """
        if not self.max_supp_attn.all:
            raise ValueError(f"No attention scores available for level {self.level}.")
        self.max_supp_attn_std.add(self.max_supp_attn.get_std())
        self.attn_on_target_std.add(self.attn_on_target.get_std())
        return self.max_supp_attn.get_mean(), self.attn_on_target.get_mean()

    def calculate_bleu(self) -> float:
        """
        Calculate the BLEU score for the sample part.

        :return: the BLEU score
        """
        for silver, pred in zip(self.silver_reasonings, self.pred_reasonings):
            bleu_score = self.bleu.bleu.compute(references=[silver], predictions=[pred])
            # Example of return
            # {'bleu': 0.07169190876271075,
            #  'precisions': [0.3, 0.1348314606741573, 0.056818181818181816, 0.011494252873563218],
            #  'brevity_penalty': 1.0,
            #  'length_ratio': 1.9148936170212767,
            #  'translation_length': 90,
            #  'reference_length': 47}
            self.bleu.add(bleu_score["bleu"])
        self.bleu_std.add(self.bleu.get_std())
        return self.bleu.get_mean()

    def calculate_rouge(self) -> float:
        """
        Calculate the ROUGE-L score for the sample part.
        It looks for the longest common subsequence between the reference and the prediction.

        :return: the ROUGE-L score
        """
        for silver, pred in zip(self.silver_reasonings, self.pred_reasonings):
            rouge_score = self.rouge.rouge.compute(
                references=[silver], predictions=[pred]
            )
            # Example of return
            # {'rouge1': np.float64(0.4786324786324786),
            #  'rouge2': np.float64(0.19130434782608696),
            #  'rougeL': np.float64(0.2564102564102564),
            #  'rougeLsum': np.float64(0.2564102564102564)}
            self.rouge.add(float(rouge_score["rougeL"]))
        self.rouge_std.add(self.rouge.get_std())
        return self.rouge.get_mean()

    def calculate_meteor(self):
        """
        Calculate the Meteor score for the sample part.

        :return: the Meteor score
        """
        for silver, pred in zip(self.silver_reasonings, self.pred_reasonings):
            meteor_score = self.meteor.meteor.compute(
                references=[silver], predictions=[pred]
            )
            # Example of return
            # {'meteor': np.float64(0.4749067889809824)}
            self.meteor.add(float(meteor_score["meteor"]))
        self.meteor_std.add(self.meteor.get_std())
        return self.meteor.get_mean()
