from __future__ import annotations

import csv
import json
import sys
import warnings
from typing import Iterable, TextIO, Union

from data.utils import *
from evaluation.Evaluator import MetricEvaluator
from inference.DataLevels import Features, Sample, SamplePart, Split, Task
from inference.Prompt import Prompt
from interpretability.utils import InterpretabilityResult
from settings.config import DataSplits


class DataSaver:
    """
    This class handles everything related to saving data.
    """

    def __init__(self, save_to: str) -> None:
        """
        Initialize the DataSaver.
        The datasaver handles everything related to saving data.

        :param save_to: the path to save the data
        """
        self.old_stdout: TextIO = sys.stdout
        # self.results_path is updated in create_result_paths
        self.results_path: Path = Path(save_to)
        self.run_path: Path = Path(save_to)
        self.prompt_name: str = ""

    def create_result_paths(
        self,
        prompt_name: str,
        splits: list[Union[DataSplits.train, DataSplits.valid, DataSplits.test]],
    ) -> tuple[str, dict[str, str], dict[str, str]]:
        """
        Create the unique results path for the run and the files to save the data:
        log file, results file, and metrics files. The results path is also updated to match the current prompt.

        :param prompt_name: the name of the prompt
        :param splits: splits of the data
        :return: the names for the log file, result files, and metric files in a tuple
        """
        self.results_path = self.run_path / prompt_name

        try:
            os.makedirs(self.results_path)
        except FileExistsError:
            print(
                f"Directory {self.results_path} already exists and is not empty. "
                "Please choose another results_path or empty the directory."
            )
            self.results_path = Path("./outputs") / prompt_name
            os.makedirs(self.results_path)
        except OSError:
            print(
                f"Creation of the directory {self.results_path} failed due "
                f"to lack of writing rights. Please check the path."
            )
            self.results_path = Path("./outputs") / prompt_name
            os.makedirs(self.results_path)

        print(f"\nThe results will be saved to {self.results_path}\n")

        results_file_names = {}
        metrics_file_names = {}

        for split in splits:
            results_file_names[split] = f"{split}_{prompt_name}_results.csv"
            metrics_file_names[split] = f"{split}_{prompt_name}_metrics.csv"

        log_file_name = f"{prompt_name}.log"

        return log_file_name, results_file_names, metrics_file_names

    def save_output(
        self,
        data: list[dict[str, str | int | float]],
        headers: list | tuple,
        file_name: str | Path,
        path_add: str = "",
        flag: str = "a+",
    ) -> None:
        """
        This function allows to save the data continuously throughout the run.
        The headers are added once at the beginning, and the data is appended
        to the end of the file.

        This is how DictWriter works:
            headers = ['first_name', 'last_name']
            row = {'first_name': 'Lovely', 'last_name': 'Spam'}
            writer.writerow(row)

        :param data: one row as list of strings or multiple such rows
        :param headers: the headers for the csv file
        :param file_name: the name of the file to save the data
        :param path_add: an addition to the results path (goes between results_path and file_name)
        :param flag: the flag to open the file
        :return: None
        """
        if isinstance(file_name, str):
            file_name = self.results_path / path_add / file_name
            Path(self.results_path / path_add).mkdir(parents=True, exist_ok=True)
            if file_name.suffix != ".csv":
                raise ValueError("The file should be saved in a .csv format.")
        else:
            file_name = Path(file_name)

        with open(file_name, flag, encoding="UTF-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t")
            if is_empty_file(file_name):
                writer.writeheader()
            [writer.writerow(row) for row in data]

    def save_split_accuracy(
        self,
        evaluators: list[MetricEvaluator],
        accuracy_file_name: str,
        multi_system: bool = False,
    ) -> None:
        """
        Save the accuracies for the split,
        including the * mean accuracy * for all tasks.

        :param evaluators: the evaluator
        :param accuracy_file_name: the name of the file to save the accuracies
        :param multi_system: if the setting uses two models
        :return: None
        """
        if multi_system:
            versions = ["before", "after"]
        else:
            versions = ["before"]

        accuracies_to_save = defaultdict(dict)
        for evaluator, version in zip(evaluators, versions):
            accuracies = list(
                format_accuracy_metrics(
                    evaluator.exact_match_accuracy,
                    evaluator.soft_match_accuracy,
                    evaluator.exact_match_std,
                    evaluator.soft_match_std,
                    version=version,
                ).values()
            )
            for acc in accuracies:
                accuracies_to_save[acc["task_id"]].update(acc)

        for acc in accuracies_to_save.values():
            self.save_output(
                data=[acc],
                headers=list(acc.keys()),
                file_name=accuracy_file_name,
            )

    def save_split_metrics(
        self, features: Features, metrics_file_name: str, version: str
    ) -> None:
        """
        Save the metrics for all the tasks in a split.

        :param features: the features to save
        :param metrics_file_name: the path to save the metrics
        :param version: the version of the features
        :return: None
        """
        headers = ["metric", "count"]
        features = [
            {h: m for h, m in zip(headers, metric)} for metric in features.get().items()
        ]
        self.save_output(
            data=features,
            headers=headers,
            file_name=metrics_file_name,
            path_add=version,
        )

    @staticmethod
    def save_with_separator(file_path: Path, data: Iterable, sep="\n") -> None:
        """
        Save with the separator between the data.

        :param file_path: the path to the file
        :param data: the data to save
        :param sep: the separator, a newline by default
        :return: None
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="UTF-8") as file:
            file.write(sep.join(map(lambda x: str(x).strip(), data)))

    @staticmethod
    def save_json(file_path: Path, data: Iterable) -> None:
        """
        Save json data

        :param file_path: the path to the file
        :param data: the data to save
        :return: None
        """
        with open(file_path, "w", encoding="UTF-8") as file:
            file.write(json.dumps(data, indent=2))

    def save_part_interpretability(
        self, result: InterpretabilityResult, version: str, part: SamplePart
    ) -> None:
        """
        Save the interpretability result per sample part.

        :param result: the interpretability result
        :param version: the version of the interpretability result (might also be an iteration)
        :param part: the part of the sample (only for the ids)
        :return: None
        """
        if not (version in ["before", "after"] or version.isdigit()):
            raise ValueError(
                "Version should be either 'before', 'after' or an iteration number."
            )
        print("Version:", version)
        if result and not result.empty():
            if version.isdigit():
                version = Path("iterations", version)
            attn_scores_subdir = (
                self.results_path / version / "interpretability" / "attn_scores"
            )
            Path.mkdir(attn_scores_subdir, exist_ok=True, parents=True)

            try:
                file_name = (
                    f"attn_scores-{part.task_id}-{part.sample_id}-{part.part_id}.txt"
                )
                print("DEBUG", result)
                print("ATTN raw type:", type(result.attn_scores))
                print(
                    "ATTN raw shape:", getattr(result.attn_scores, "shape", "no shape")
                )
                print(
                    "DEBUG result.interpretability.attn_scores",
                    result.attn_scores.size,
                    "\n",
                    result.attn_scores,
                )
                if result.attn_scores.size != 0:
                    attn_scores = [
                        "\t".join(map(str, row)) for row in result.attn_scores.tolist()
                    ]
                    self.save_with_separator(
                        file_path=attn_scores_subdir / file_name, data=attn_scores
                    )
                else:
                    warnings.warn(
                        f"No attention scores found for task {part.task_id}, sample {part.sample_id}, part {part.part_id}."
                    )
                for tokens in ("x_tokens", "y_tokens"):
                    if not hasattr(result, tokens):
                        warnings.warn(
                            f"No {tokens} for task {part.task_id}, sample {part.sample_id}, part {part.part_id}."
                        )
                        continue
                    file_name = (
                        f"{tokens}-{part.task_id}-{part.sample_id}-{part.part_id}.txt"
                    )
                    self.save_with_separator(
                        file_path=attn_scores_subdir / file_name,
                        data=getattr(result, tokens),
                    )

            except AttributeError as e:
                print(f"AttributeError: {e} in {result}")

                print(
                    f"Interpretability results for task {part.task_id} saved to {attn_scores_subdir}"
                )
        else:
            warnings.warn("No interpretability results found and saved.")

    def save_interpretability(self, task_data: Task) -> None:
        """
        Save the interpretability result per sample part.

        :param task_data: the task instance with the results
        :return: None
        """
        for part in task_data.parts:
            for result, version in zip(part.results, part.versions):
                self.save_part_interpretability(result.interpretability, version, part)

    def save_feedback_iteration(
        self,
        part: SamplePart,
        iteration: int,
        student_message: str,
        teacher_message: str,
        interpretability: InterpretabilityResult = None,
    ) -> None:
        """
        Save both student's output and teacher's feedback in a single operation.

        :param part: The SamplePart object containing IDs
        :param iteration: The iteration number
        :param student_message: The student's message
        :param teacher_message: The teacher's feedback
        :param interpretability: The interpretability result
        """
        iterations_dir = self.results_path / "iterations"
        iterations_dir.mkdir(exist_ok=True, parents=True)

        part_identifier = f"{part.task_id}-{part.sample_id}-{part.part_id}"

        # Save both messages in one operation (open files only once)
        student_file = iterations_dir / f"{iteration}_student_{part_identifier}.txt"
        teacher_file = iterations_dir / f"{iteration}_teacher_{part_identifier}.txt"

        with open(student_file, "w", encoding="UTF-8") as sf, open(
            teacher_file, "w", encoding="UTF-8"
        ) as tf:
            sf.write(student_message)
            tf.write(teacher_message)

        if interpretability:
            self.save_part_interpretability(interpretability, str(iteration), part)

    def save_part_result(self, part: SamplePart) -> None:
        """
        Save the result of a part of a sample.

        :param part: the part of the sample
        :return: None
        """
        result = part.get_result()
        self.save_output(
            data=[result],
            headers=list(result.keys()),
            file_name=f"t_{part.task_id}_s_{part.sample_id}_results.csv",
        )
        for result, version in zip(part.results, part.versions):
            self.save_part_interpretability(result.interpretability, version, part)

    def save_sample_result(self, sample: Sample) -> None:
        """
        Save the results for the task and the accuracy for the task to the separate files.

        :param sample: the result of the task
        :return: None
        """
        for part in sample.parts:
            self.save_part_result(part)

    def save_task_result(
        self,
        task_id: int,
        task_data: Task,
        headers: list[str],
        results_file_name: str,
        metrics_file_name: str,
    ) -> None:
        """
        Save the results for the task and the accuracy for the task to the separate files.

        :param task_id: the task id
        :param task_data: the result of the task
        :param headers: the headers for the results
        :param results_file_name: the name of the file to save the results specific to the split
        :param metrics_file_name: the name of the file to save the accuracy specific to the split
        :return: None
        """
        self.save_output(
            data=task_data.results,
            headers=headers,
            file_name=results_file_name,
        )
        # get accuracy for the last task
        task_accuracy = {
            "task_id": task_id,
            **task_data.evaluator_after.get_accuracies(),
        }
        self.save_interpretability(task_data)
        self.save_output(
            data=[task_accuracy],
            headers=list(task_accuracy.keys()),
            file_name=metrics_file_name,
            path_add="",
        )

    def save_run_accuracy(
        self,
        task_ids: list[int],
        splits: dict[Prompt, Split],
        split_name: str,
        version: str = "after",
    ) -> None:
        """
        Save the accuracies for the split run, including the mean accuracy for all tasks.

        :param task_ids: the task ids
        :param splits: the split objects of the data
        :param split_name: the name of the split
        :param version: if to save the accuracy for after the setting was applied
        :return: None
        """
        run_metrics = {}
        run_headers = ["task_id"]

        for prompt, split in splits.items():
            prompt_headers = prepare_accuracy_headers(prompt.name, version=version)

            if version == "after":
                evaluator = split.evaluator_after
            elif version == "before":
                evaluator = split.evaluator_before
            else:
                raise ValueError(
                    f"Version should be either 'before' or 'after', not {version}."
                )

            run_metrics = format_task_accuracies(
                accuracies_to_save=run_metrics,
                task_ids=task_ids,
                exact_match_accuracies=evaluator.exact_match_accuracy,
                soft_match_accuracies=evaluator.soft_match_accuracy,
                exact_match_std=evaluator.exact_match_std,
                soft_match_std=evaluator.soft_match_std,
                headers=prompt_headers,
            )
            if version == "after":
                features = split.features_after
            elif version == "before":
                features = split.features_before
            else:
                raise ValueError("Version should be either 'before' or 'after'.")

            run_metrics = format_split_metrics(
                features, prompt_headers, run_metrics, version
            )
            run_headers.extend(prompt_headers.values())

        mean_headers = prepare_accuracy_headers("mean")
        run_headers.extend(mean_headers.values())
        run_metrics = calculate_mean_accuracies(run_metrics, mean_headers, version)

        self.save_output(
            data=list(run_metrics.values()),
            headers=run_headers,
            file_name=self.run_path / f"{split_name}_accuracies.csv",
            path_add=version,
        )

    def redirect_printing_to_log_file(self, file_name: str) -> TextIO:
        """
        Allows to redirect printing during the script run from console into a log file.
        Old 'sys.stdout' that must be returned in place after the run by calling
        DataHandler.return_console_printing!

        :param file_name: the path and name of the file to redirect the printing
        :return: log file to write into
        """
        log_file = open(self.results_path / file_name, "w", encoding="UTF-8")
        sys.stdout = log_file
        return log_file

    def return_console_printing(self, file_to_close: TextIO) -> None:
        """
        This function allows to return the console printing to the original state.

        :param file_to_close: the file that should be closed
        :return:
        """
        file_to_close.close()
        sys.stdout = self.old_stdout
