from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from data.Statistics import Statistics
from prompts.Chat import Chat, Source
from prompts.Prompt import Prompt
from settings.Model import Model
from settings.utils import Enumerate


class Setting(ABC):
    """
    Abstract class for settings
    """

    @abstractmethod
    def __init__(
        self,
        model: Model,
        to_enumerate: dict[Enumerate, bool],
        parse_output: bool,
        statistics: Statistics,
        prompt: Prompt = None,
        samples_per_task: int = 5,
    ):
        """
        Baseline class manages model runs and data flows around it.
        It is intended to use in the further settings.

        :param parse_output: if we want to parse the output of the model (currently looks for 'answer' and 'reasoning')
        :param statistics: class for statistics
        :param prompt: system prompt to start conversations
        :param samples_per_task: number of samples per task for logging
        """
        self.model = model

        self.prompt = prompt
        self.to_enumerate = to_enumerate
        self.parse_output = parse_output

        self.stats = statistics

        self.question_id = 0
        self.total_samples = 0
        self.total_tasks = 0
        self.total_parts = 0
        self.samples_per_task = samples_per_task

        self.accuracies_per_task: list = []
        self.soft_match_accuracies_per_task: list = []

        self.accuracy: int = 0
        self.soft_match_accuracy: int = 0

    @abstractmethod
    def prepare_prompt(self, sample_part: dict[str, dict], chat: Chat) -> str:
        """
        Prepares the prompt to include the current part of the sample.
        :param sample_part:
        :param chat:
        :return: prompt with the task and the current part
        """
        raise NotImplementedError

    @abstractmethod
    def apply_setting(self, decoded_output: str) -> dict[str, str]:
        """
        Apply setting-specific postprocessing of the inital model output.
        For the baseline and skyline, this consists of parsing the output.
        For the SD and feedback setting, this entails the main idea of these settings.

        :param decoded_output: the decoded output
        :return: parsed output
        """
        # ALSO INCLUDES SETTINGS -> SD AND FEEDBACK
        raise NotImplementedError

    def iterate_task(
        self,
        task_id: int,
        task_data: dict[int, list[dict[str, dict]]],
        prompt_name: str,
    ) -> list[dict[str, int | str]]:
        """
        Manages the data flow in and out of the model, iteratively going through
        each part of each sample in a given task, recoding the results to report.

        The following steps apply:
        1. iterate through tasks
        2. reformat the data into parts
           Hint: a part is defined as a list with context sentences finished with
                 a question.
        3. iterate through parts
        4. create and format the prompt
        5. call the model and yield the response
        6. add the model's output to conversation
        7. report the results for a sample: answers and accuracy
        8. report the results for the task:  accuracy

        :param task_id: task id corresponding to task name in original dataset
        :param task_data: task data of the following structure:
        {
            sample_id: str = 0-n:
            [ # there might be multiple parts for one sample
                 {
                     "context": {
                         line_num: str
                         sentence: str
                     }
                     "question": {
                         line_num: str
                         question: str
                     }
                     "answer": {
                         line_num: str
                         answers: list[str]
                     }
                     "supporting_fact": [
                         [int], [int, int]
                     ]
                 }
             ]
         }
        :param prompt_name: name of the prompt
        :return: results for the task in a list of dicts with each dict representing
                 one call to the model and will end up as one row of the table
        """
        task_results = []
        sample_accuracies = []
        sample_soft_match_accuracies = []

        # 1. Iterate through samples
        for sample_id, sample_parts in list(task_data.items()):
            # each sample is a new conversation
            chat = Chat(system_prompt=self.prompt.text)
            # collect the true answers
            y_true_sample = [
                ", ".join(list(part["answer"].values())[0]) for part in sample_parts
            ]
            y_pred_sample = []
            sample_id_ = sample_id + 1
            part_id = 0

            # 2. Iterate through parts (one question at a time)
            for sample_part, y_true in zip(sample_parts, y_true_sample):
                self.question_id += 1
                part_id += 1
                print(
                    "\n-* "
                    f"TASK {task_id}/{self.total_tasks} | "
                    f"SAMPLE {sample_id_}/{self.samples_per_task} | "
                    f"PART {part_id}/{len(sample_parts)} | "
                    f"{prompt_name} | "
                    f"RUN ID {self.question_id}/{self.total_parts} "
                    "*-",
                    end="\n\n\n",
                )

                formatted_part = self.prompt.format_part(
                    part=sample_part, to_enumerate=self.to_enumerate
                )

                chat.add_message(part=formatted_part, source=Source.user)

                formatted_prompt = self.prepare_prompt(
                    sample_part=sample_part, chat=chat
                )

                # 5. Call the model and yield the response
                decoded_output = self.model.call(prompt=formatted_prompt)
                print(
                    "Model's output:",
                    decoded_output,
                    end="\n\n\n",
                    flush=True,
                )

                # 6. Add the model's output to conversation
                chat.add_message(part=decoded_output, source=Source.assistant)

                model_output = self.apply_setting(decoded_output=decoded_output)

                part_result = {
                    "id": self.question_id,
                    "task_id": task_id,
                    "sample_no": sample_id_,
                    "task": formatted_part,
                    "true_result": y_true,
                }
                part_result.update(model_output)

                task_results.append(part_result)

                y_pred_sample.append(decoded_output)

            # 7. Report the results for the sample: answers and accuracy
            print(
                "Model's predictions for the sample:",
                "\t{:<18s} PREDICTED".format("GOLDEN"),
                sep="\n\n",
                end="\n\n",
            )
            [
                print(
                    "\t{0:<18s} {1}".format(true, predicted.replace("\n", "\t")),
                )
                for true, predicted in zip(y_true_sample, y_pred_sample)
            ]

            sample_accuracy = round(
                self.stats.accuracy_score(y_true_sample, y_pred_sample), 2
            )
            sample_accuracies.append(sample_accuracy)
            print(
                f"\nAccuracy score for sample {sample_id_}:",
                sample_accuracy,
            )

            sample_soft_match_accuracy = round(
                self.stats.soft_match_accuracy_score(y_true_sample, y_pred_sample), 2
            )
            sample_soft_match_accuracies.append(sample_soft_match_accuracy)
            print(
                f"Soft accuracy for sample {sample_id_}:",
                sample_soft_match_accuracy,
                end="\n\n\n",
            )

        # 8. Report the results for the task: accuracy
        print("\n- TASK RESULTS -", end="\n\n")

        task_accuracy = round(sum(sample_accuracies) / len(sample_accuracies), 2)
        self.accuracies_per_task.append(task_accuracy)

        print(f"Accuracy score for task {task_id}:", task_accuracy)
        task_results[0]["accuracy"] = task_accuracy

        task_soft_match_accuracy = round(
            sum(sample_soft_match_accuracies) / len(sample_soft_match_accuracies), 2
        )
        self.soft_match_accuracies_per_task.append(task_soft_match_accuracy)

        print(
            f"Soft match accuracy for task {task_id}:",
            task_soft_match_accuracy,
            end="\n\n",
        )

        task_results[0]["soft_match_accuracy"] = task_soft_match_accuracy

        print(f"The work on task {task_id} is finished successfully")

        # Clear the cache
        torch.cuda.empty_cache()

        return task_results
