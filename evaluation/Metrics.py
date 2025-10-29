from __future__ import annotations

import statistics
import warnings

import evaluate
import numpy as np


class Metric:
    def __init__(self, name: str, var: str, values: list[float] = None):
        """
        Initialize the metric class.

        :param name: the name of the metric
        :param var: the variable name
        :param values: the list of metric values
        """
        self.name: str = name
        self.var: str = var
        self.all: list[float] = values if values else []

        self.mean: float = None
        self.std: float = None

    def __repr__(self):
        """
        Return a string representation of the metric.
        """
        return f"{self.name} Metric: {self.get_mean()} ± {self.get_std()}"

    def __getitem__(self, slice_):
        """
        Support int, slice, and numpy.ndarray indexing.
        """
        if isinstance(slice_, np.ndarray):
            # NumPy array of indices
            # Convert to a list of ints, then use standard Python indexing
            return [self.all[int(i)] for i in slice_]
        elif isinstance(slice_, (list, tuple)):
            # Allow list or tuple of indices (assuming (index,) is at position 0)
            return [self.all[i[0]] for i in slice_]
        else:
            # Normal indexing or slicing
            return self.all[slice_]

    def __iter__(self) -> iter:
        """
        Return an iterator over all metric values.
        """
        return iter(self.all)

    def __len__(self) -> int:
        """
        Return the number of metric values.
        """
        return len(self.all)
    
    def map_to_float(self, value: str) -> float:
        """
        Map string values to float.

        :param value: the string value
        :return: the mapped float value
        """
        mapping = {
            "fully": 1.0,
            "partially": 0.5,
            "none": 0.0
        }
        return mapping.get(value.lower(), 0.0)

    def add(self, metric: Metric | int | float | np.float64 | list[int | float] | None) -> None:
        """
        Add metric values to the list of all metric values.

        :param metric: the metric values
        """
        type_ = type(metric)
        if type_ is Metric or issubclass(type_, Metric):
            self.all.append(metric.get_mean())
        elif type_ in [int, float, np.float64]:
            self.all.append(metric)
        elif type_ is bool:
            self.all.append(int(metric))
        elif type_ is str:
            try:
                value = self.map_to_float(metric)
            except KeyError:
                raise KeyError(
                    f"Found invalid string metric value: {metric}"
                )
            self.all.append(value)
        elif type_ is list:
            for m in metric:
                if type(m) in [int, float, np.float64]:
                    self.all.append(m)
                elif isinstance(m, str) and m.lower() == "nan":
                    self.all.append(0.0)
                    warnings.warn(
                        f"Found 'nan' in metric values, treating as 0.0: {metric}"
                    )
                else:
                    warnings.warn(
                        f"Found invalid metric value (skipping): {type(m)} {m}"
                    )
        else:
            raise TypeError(
                f"{self.name} must be a float or a list of floats, not {type_}"
            )

    def get_mean(self) -> float:
        """
        Return the mean of the metric values.
        """
        if len(self.all) == 0:
            self.mean = 0.0
        else:
            self.mean = round(statistics.mean(self.all), 2)
        return self.mean

    def get_std(self) -> float:
        """
        Return the standard deviation of the metric values.
        """
        if len(self.all) < 2:
            self.std = 0.0
        else:
            self.std = round(statistics.stdev(self.all), 2)
        return self.std


class Accuracy(Metric):
    def __init__(self, name: str, var: str, accuracies: list[float] = None):
        """
        Initialize the accuracy class.

        :param name: the type of accuracy
        :param var: the variable name
        :param accuracies: the list of accuracy values
        """
        super().__init__(name, var, accuracies)


class AttnDistribution(Metric):
    def __init__(self, name: str, var: str, max_supp_target: list[float] = None):
        """
        Initialize the attention distribution class for tracking the ratio of max attention on target tokens.

        :param name: the type of attention distribution
        :param var: the variable name
        :param max_supp_target: the list of attention distribution values
        """
        super().__init__(name, var, max_supp_target)


class Correlation(Metric):
    """
    Class for tracking the correlation between two metrics.
    :param name: the type of correlation
    :param var: the variable name
    :param correlations: the list of correlation values
    :param p_values: the list of correlation p-values
    """

    def __init__(
        self,
        name,
        var: str,
        correlations: list[float] = None,
        p_values: list[float] = None,
    ):
        super().__init__(name, var, correlations)
        self.p_values: list[float] = p_values if p_values else []


class AttnOnTarget(Metric):
    def __init__(self, name: str, var: str, attn_on_target: list[float] = None):
        """
        Initialize the class for tracking the attention on target tokens.

        :param name: the type of attention
        :param var: the variable name
        :param attn_on_target: the list of attention values on target tokens
        """
        super().__init__(name, var, attn_on_target)


class BLEU(Metric):
    def __init__(self, name: str, var: str, scores: list[float] = None):
        """
        Initialize the BLEU class.

        :param name: the type of BLEU
        :param var: the variable name
        :param scores: the list of BLEU values
        """
        super().__init__(name, var, scores)
        self.bleu = evaluate.load("bleu")


class ROUGE(Metric):
    def __init__(self, name: str, var: str, scores: list[float] = None):
        """
        Initialize the ROUGE class.

        :param name: the type of ROUGE
        :param var: the variable name
        :param scores: the list of ROUGE values
        """
        super().__init__(name, var, scores)
        self.rouge = evaluate.load("rouge")


class Meteor(Metric):
    def __init__(self, name: str, var: str, scores: list[float] = None):
        """
        Initialize the Meteor class.

        :param name: the type of Meteor
        :param var: the variable name
        :param scores: the list of Meteor values
        """
        super().__init__(name, var, scores)
        self.meteor = evaluate.load("meteor")
