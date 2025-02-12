import os
from pathlib import Path


class Task:
    """
    A class to represent a task.
    """

    def __init__(
        self,
        number: int,
        to_enumerate: bool = True,
        handpicked: bool = False,
        not_mentioned: bool = False,
    ):
        """
        Initialize the task. The task can be enumerated or not enumerated,
        handpicked or not handpicked, and augmented with not_mentioned examples or not.
        """
        self.number = number
        self.to_enumerate = to_enumerate
        self.folder = (
            Path("data/examples") / "enumerated" if to_enumerate else "not_enumerated"
        )

        self.handpicked = handpicked
        if self.handpicked and not_mentioned:
            self.folder = self.folder / "handpicked_aug"
        elif self.handpicked:
            self.folder = self.folder / "handpicked"
        else:
            self.folder = self.folder / "from_valid"

    def __repr__(self):
        raise NotImplementedError(
            "The __repr__ method is not implemented, use TaskExample class."
        )

    def __iter__(self):
        raise NotImplementedError(
            "The __iter__ method is not implemented, use TaskExamples class."
        )


class TaskExample(Task):
    """
    A class to represent an example for a task.
    """

    def __repr__(self):
        """
        Return the one example for the task.
        """
        paths = [
            i
            for i in self.folder.iterdir()
            if i.name.startswith(f"task_{self.number}_")
        ]
        with open(paths[0], "r", encoding="utf-8") as file:
            return file.read()


class TaskExamples(Task):
    """
    A class to represent all examples for a task.
    """

    def __iter__(self):
        """
        Return an iterator over all the examples for the task.
        """
        if self.handpicked:
            raise NotImplementedError(
                "Getting multiple handpicked examples is not implemented, use TaskExample class."
            )

        all_files = [
            file
            for file in os.listdir(self.folder)
            if file.startswith(f"task_{self.number}_")
        ]
        all_examples = []
        for i, file in enumerate(all_files, 1):
            path = self.folder / f"task_{self.number}_example_{i}.txt"
            with open(path, "r", encoding="utf-8") as file:
                all_examples.append(file.read())
        return iter(all_examples)


if __name__ == "__main__":
    # to print one example for the task
    print("Random example:", end="\n\n")
    print(TaskExample(number=19))

    print("\nHandpicked example:", end="\n\n")
    print(TaskExample(number=19, handpicked=True))

    print("\nHandpicked example and augmented:", end="\n\n")
    print(TaskExample(number=19, handpicked=True, not_mentioned=True))

    print("\nAll examples:", end="\n\n")
    for example in TaskExamples(number=19):
        print(example, end="\n\n")
