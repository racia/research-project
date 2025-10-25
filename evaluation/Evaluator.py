from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Union

from evaluation.Metrics import Accuracy, Correlation, BLEU, Meteor, Metric, ROUGE
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

        self.max_supp_attn_corr: Correlation = Correlation(
            f"Correlation of Accuracy with Max Attn Distribution {self.version.capitalize()}",
            "max_supp_attn_corr",
        )

        self.attn_on_target_corr: Correlation = Correlation(
            f"Correlation of Accuracy with Attn on Target Distribution {self.version.capitalize()}",
            "attn_on_target_corr",
        )

        self.sample_part_lengths_corr: Correlation = Correlation(
            f"Correlation of Accuracy with Sample Part Lengths {self.version.capitalize()}",
            "sample_part_lengths_corr",
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
            f"meteor_std={self.meteor.get_std()}>,"
            f"max_supp_attn_corr={self.max_supp_attn_corr.get_mean()}>, "
            f"attn_on_target_corr={self.attn_on_target_corr.get_mean()}>, "
            f"sample_part_lengths_corr={self.sample_part_lengths_corr.get_mean()}>, "
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

        self.parts_answer_correct = Metric(
            "Parts Answer Correct", "part_answer_correct"
        )
        self.parts_max_supp_attn = Metric("Parts Max Supp Attn", "part_max_supp_attn")
        self.parts_attn_on_target = Metric(
            "Parts Attn on Target", "part_attn_on_target"
        )
        self.ids_with_bleu = defaultdict(dict)
        self.ids_with_rouge = defaultdict(dict)
        self.ids_with_meteor = defaultdict(dict)

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

    def get_metrics(
        self, as_lists: bool = False
    ) -> dict[str, Accuracy | Metric | float]:
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
                f"max_supp_attn_{self.version}": self.max_supp_attn,
                f"max_attn_dist_std_{self.version}": self.max_supp_attn_std,
                f"max_attn_on_target_{self.version}": self.attn_on_target,
                f"attn_on_target_std_{self.version}": self.attn_on_target_std,
                f"bleu_{self.version}": self.bleu,
                f"bleu_std_{self.version}": self.bleu_std,
                f"rouge_{self.version}": self.rouge,
                f"rouge_std_{self.version}": self.rouge_std,
                f"meteor_{self.version}": self.meteor,
                f"meteor_std_{self.version}": self.meteor_std,
                f"max_supp_attn_corr_{self.version}": self.max_supp_attn_corr,
                f"attn_on_target_corr_{self.version}": self.attn_on_target_corr,
                f"sample_part_lengths_corr_{self.version}": self.sample_part_lengths_corr,
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
            f"max_supp_attn_corr_{self.version}": self.max_supp_attn_corr.get_mean(),
            f"attn_on_target_corr_{self.version}": self.attn_on_target_corr.get_mean(),
            f"sample_part_lengths_corr_{self.version}": self.sample_part_lengths_corr.get_mean(),
            f"max_supp_attn_corr_std_{self.version}": self.max_supp_attn_corr.get_std(),
            f"attn_on_target_corr_std_{self.version}": self.attn_on_target_corr.get_std(),
            f"sample_part_lengths_corr_std_{self.version}": self.sample_part_lengths_corr.get_std(),
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
                f"max_supp_attn_{self.version}": self.max_supp_attn,
                f"max_supp_attn_std_{self.version}": self.max_supp_attn_std,
                f"attn_on_target_{self.version}": self.attn_on_target,
                f"attn_on_target_std_{self.version}": self.attn_on_target_std,
            }
        return {
            f"max_supp_attn_{self.version}": self.max_supp_attn.get_mean(),
            f"max_supp_attn_std_{self.version}": self.max_supp_attn.get_std(),
            f"attn_on_target_{self.version}": self.attn_on_target.get_mean(),
            f"attn_on_target_std_{self.version}": self.attn_on_target.get_std(),
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

    def calculate_correlation(
        self,
        **kwargs: Union[list[float | int | bool], str],
    ) -> dict:
        """
        Calculate the correlation score between each arg metric1 scores list with each kwarg metric2 scores list on task level.
        :param kwargs: One or more str metric or list of scores, e.g. max_supp_attn or attn_on_target scores
        :return: correlation matrix of the metrics
        """
        for key, value in kwargs.items():
            kwargs[key] = getattr(self, value).all if isinstance(value, str) else value

        corr_matrix = defaultdict(dict)
        for base_name, base_values in kwargs.items():
            for add_name, add_values in kwargs.items():
                assert (
                    base_values
                    and isinstance(base_values, list)
                    and isinstance(base_values[0], (float, int, bool))
                ), f"The base values must be a list of floats, integers or booleans, given {base_values}."
                assert (
                    add_values
                    and isinstance(add_values, list)
                    and isinstance(add_values[0], (float, int, bool))
                ), f"The base values must be a list of floats, integers or booleans, given {add_values}."
                assert len(base_values) == len(add_values), (
                    f"Length of {base_values} ({len(base_values)}) and {add_values} ({len(add_values)}) "
                    f"must be equal for correlation calculation."
                )
                corr_score, p_value = self.stats.corr_score(base_values, add_values)
                corr_matrix[base_name][add_name] = corr_score, p_value
                var = f"{base_name}_{add_name}_corr"
                name = f"Correlation of {base_name} with {add_name} {self.version.capitalize()}"
                corr = Correlation(name, var)
                corr.add(corr_score)
                corr.p_values.append(round(p_value, 2))
                setattr(self, var, corr)

        return corr_matrix

    def get_correlations(self, as_lists: bool = False) -> dict[str, float | Metric]:
        """
        Get the correlation scores for the evaluator.

        :return: the correlation scores
        """
        if as_lists:
            return {
                f"max_supp_attn_corr_{self.version}": self.max_supp_attn_corr,
                f"max_supp_att_corr_std{self.version}": self.max_supp_attn_corr.get_std(),
                f"attn_on_target_corr_{self.version}": self.attn_on_target_corr,
                f"attn_on_target_corr_std_{self.version}": self.attn_on_target_corr.get_std(),
            }
        return {
            f"max_supp_attn_corr_{self.version}": self.max_supp_attn_corr.get_mean(),
            f"max_supp_attn_corr_std_{self.version}": self.max_supp_attn_corr.get_std(),
            f"attn_on_target_corr_{self.version}": self.attn_on_target_corr.get_mean(),
            f"attn_on_target_corr_std_{self.version}": self.attn_on_target_corr.get_std(),
        }


class AnswerEvaluator(MetricEvaluator):
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
        attn_metrics = (self.attn_on_target.all, self.max_supp_attn.all)
        if not all(attn_metrics):
            raise ValueError(
                f"No attention scores available for level {self.level}: "
                f"{attn_metrics}."
            )
        if any(type(attn) is not list for attn in attn_metrics):
            raise TypeError(
                f"Expected list of attention scores, got "
                f"{type(self.max_supp_attn.all)} ({self.max_supp_attn.all}) "
                f"and {type(self.attn_on_target.all)} ({self.attn_on_target.all})"
            )
        self.max_supp_attn_std.add(self.max_supp_attn.get_std())
        self.attn_on_target_std.add(self.attn_on_target.get_std())
        return self.max_supp_attn.get_mean(), self.attn_on_target.get_mean()

    def calculate_bleu(self) -> float:
        """
        Calculate the BLEU score for the sample part.

        :return: the BLEU score
        """
        for silver, pred in zip(self.silver_reasonings, self.pred_reasonings):
            if not all((silver, pred)):
                self.bleu.add(0.0)
                warnings.warn(
                    f"Both silver reasoning and prediction must be provided for BLEU calculation. "
                    f"Got silver: '{silver}', pred: '{pred}'."
                )
                continue
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
            if not all((silver, pred)):
                self.rouge.add(0.0)
                warnings.warn(
                    f"Both silver reasoning and prediction must be provided for ROUGE calculation. "
                    f"Got silver: '{silver}', pred: '{pred}'."
                )
                continue
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
            if not all((silver, pred)):
                self.meteor.add(0.0)
                warnings.warn(
                    f"Both silver reasoning and prediction must be provided for METEOR calculation. "
                    f"Got silver: '{silver}', pred: '{pred}'."
                )
                continue
            meteor_score = self.meteor.meteor.compute(
                references=[silver], predictions=[pred]
            )
            # Example of return
            # {'meteor': np.float64(0.4749067889809824)}
            self.meteor.add(float(meteor_score["meteor"]))
        self.meteor_std.add(self.meteor.get_std())
        return self.meteor.get_mean()
