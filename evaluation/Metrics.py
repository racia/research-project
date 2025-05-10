from __future__ import annotations

import statistics


class Metric:
    def __init__(self, name: str, values: list[float] = None):
        """
        Initialize the metric class.

        :param name: the name of the metric
        :param values: the list of metric values
        """
        self.name: str = name
        self.all: list[float] = values if values else []

        self.mean: float = None
        self.std: float = None

    def __getitem__(self, slice_: slice) -> float | list[float]:
        """
        Return the metric at the given index.
        """
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

    def add(self, metric: Metric | float | list[float] | None) -> None:
        """
        Add metric values to the list of all metric values.

        :param metric: the metric values
        """
        if metric:
            type_ = type(metric)
            if type_ is Metric or issubclass(type_, Metric):
                self.all.append(metric.get_mean())
            elif type_ is float:
                self.all.append(metric)
            elif type_ is list:
                self.all.extend(metric)
            else:
                raise TypeError(f"{self.name} must be a float or a list of floats.")

    def get_mean(self) -> float:
        """
        Return the mean of the metric values.
        """
        if len(self.all) == 0:
            return 0.0
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
    def __init__(self, name: str, accuracies: list[float] = None):
        """
        Initialize the accuracy class.

        :param name: the type of accuracy
        :param accuracies: the list of accuracy values
        """
        super().__init__(name, accuracies)


class AttnDistribution(Metric):
    def __init__(self, name: str, max_supp_target: list[float] = None):
        """
        Initialize the attention distribution class for tracking the ratio of max attention on target tokens.

        :param name: the type of attention distribution
        :param max_supp_target: the list of attention distribution values
        """
        super().__init__(name, max_supp_target)


class AttnOnTarget(Metric):
    def __init__(self, name: str, attn_on_target: list[float] = None):
        """
        Initialize the class for tracking the attention on target tokens.

        :param name: the type of attention
        :param attn_on_target: the list of attention values on target tokens
        """
        super().__init__(name, attn_on_target)
