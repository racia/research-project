from __future__ import annotations

import statistics


class Accuracy:
    def __init__(self, type_: str, accuracies: list[float] = None):
        """
        Initialize the accuracy class.

        :param type_: the type of accuracy
        """
        self.type = type_
        self.name = (
            f"{'-'.join([t.capitalize() for t in self.type.split('_')])} Accuracy"
        )

        self.all = accuracies if accuracies else []

        self.std = None
        self.mean = None

    def __getitem__(self, slice_: slice) -> float | list[float]:
        """
        Return the accuracy at the given index.
        """
        return self.all[slice_]

    def __iter__(self) -> iter:
        """
        Return an iterator over all accuracies.
        """
        return iter(self.all)

    def __len__(self) -> int:
        """
        Return the number of accuracies.
        """
        return len(self.all)

    def add(self, accuracy: Accuracy | float | list[float]) -> None:
        """
        Add accuracies to the list of accuracies.

        :param accuracy: the accuracies
        """
        if type(accuracy) == Accuracy:
            self.all.append(accuracy.get_mean())
        elif type(accuracy) == float:
            self.all.append(accuracy)
        elif type(accuracy) == list:
            self.all.extend(accuracy)
        else:
            raise TypeError("Accuracy must be a float or a list of floats.")

    def get_mean(self) -> float:
        """
        Return the mean of the accuracies.
        """
        if len(self.all) == 0:
            return 0.0
        self.mean = round(statistics.mean(self.all), 2)
        return self.mean

    def get_std(self) -> float:
        """
        Return the standard deviation of the accuracies.
        """
        if len(self.all) < 2:
            self.std = 0.0
        else:
            self.std = round(statistics.stdev(self.all), 2)
        return self.std
