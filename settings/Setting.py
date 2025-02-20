
from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
import torch

from data.Statistics import Statistics
from interpretability.Interpretability import Interpretability
from prompts.Chat import Chat, Source
from prompts.Prompt import Prompt
from settings.Model import Model
from settings.baseline import utils
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
        samples_per_task: int = -1,
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
        self.total_tasks = 0
        self.total_samples = 0
        self.total_parts = 0
        self.samples_per_task = samples_per_task

        self.accuracies_per_task: list = []
        self.soft_match_accuracies_per_task: list = []

        self.accuracy: int = 0
        self.soft_match_accuracy: int = 0

    @abstractmethod
    def prepare_prompt(self, sample_part: dict[str, dict], chat: Chat) -> str:
        """
        @TODO: self.to_continue check, default: add_generation_prompt
        Prepares the prompt to include the current part of the sample.
        :param sample_part:
        :param chat:
        :return: prompt with the task and the current part
        """
        raise NotImplementedError

    @abstractmethod
    def apply_setting(self, decoded_output: str, fine_tue: bool = False) -> dict[str, str]:
        """
        Apply setting-specific postprocessing of the inital model output.
        For the baseline and skyline, this consists of parsing the output.
        For the SD and feedback setting, this entails the main idea of these settings.

        :param decoded_output: the decoded output
        :return: parsed output
        """
        # ALSO INCLUDES SETTINGS -> SD AND FEEDBACK
        raise NotImplementedError

    @staticmethod
    def print_sample_predictions(trues, preds):
        """
        Print the model's predictions to compare with true values.

        :param trues: list of true values
        :param preds: list of predicted values
        """
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
            for true, predicted in zip(trues, preds)
        ]

    def calculate_and_print_accuracies(
        self,
        id_: int,
        what: str,
        trues_preds: tuple[list[str], list[str]] = None,
        str_acc_soft_acc: tuple[list[float], list[float]] = None,
        to_print: bool = True,
    ) -> tuple[float, float]:
        """
        Calculate the accuracy scores for the sample and print them.

        :param id_: id of the sample
        :param what: what is being evaluated
        :param trues_preds: true and predicted values
        :param str_acc_soft_acc: strict and soft-match accuracies
        :param to_print: whether to print the results
        :return: accuracy, soft-match accuracy
        """
        accuracy, soft_match_accuracy = 0.0, 0.0
        if (trues_preds and str_acc_soft_acc) or not (trues_preds or str_acc_soft_acc):
            raise ValueError(
                "The function requires either true and predicted values or strict and soft-match accuracies."
            )

        if trues_preds:
            trues, preds = trues_preds
            accuracy = self.stats.accuracy_score(trues, preds)
            soft_match_accuracy = self.stats.soft_match_accuracy_score(trues, preds)

        elif str_acc_soft_acc:
            accuracies, soft_match_accuracies = str_acc_soft_acc
            accuracy = statistics.mean(accuracies)
            soft_match_accuracy = statistics.mean(soft_match_accuracies)

        accuracy = round(accuracy, 2)
        soft_match_accuracy = round(soft_match_accuracy, 2)

        if to_print:
            print(
                f"\nAccuracy score for {what} {id_}:",
                accuracy,
            )
            print(
                f"Soft-match accuracy score for {what} {id_}:",
                soft_match_accuracy,
                end="\n\n\n",
            )
        return accuracy, soft_match_accuracy

    def iterate_task(
        self,
        task_id: int,
        task_data: dict[int, list[dict[str, dict]]],
        prompt_name: str,
        interpr: Interpretability = None
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
        7. Parse output (and save for fine-tuning)
        8. Call interpretability attention score method
        9. report the results for a sample: answers and accuracy
        10. report the results for the task:  accuracy

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

        if not interpr.switch:
            interpr = None
        else:
            # Initialize interpretability for each new sample - since new chat
            interpretability1 = Interpretability(self.model, self.model.tokenizer, task_id)

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

                chat.add_message(part=formatted_part, source=Source.user, interpretability=interpr)

                formatted_prompt = self.prepare_prompt(
                    sample_part=sample_part, chat=chat
                )

                # 5. Call the model and yield the response
                decoded_output = self.model.call(prompt=formatted_prompt)
                print(
                    "Model's output:",
                    decoded_output.lower(),
                    end="\n\n\n",
                    flush=True,
                )

                # 6. Add the model's output to conversation
                chat.add_message(part=decoded_output, source=Source.assistant, interpretability=interpr)

                # 7. Parse output (and if fune-tuning: write in task_id files)
                model_output = self.apply_setting(decoded_output=decoded_output, fine_tune=False, task_id=task_id, formatted_part=formatted_part)

                part_result = {
                    "part_id": part_id,
                    "id": self.question_id,
                    "task_id": task_id,
                    "sample_no": sample_id_,
                    "task": formatted_part,
                    "true_result": y_true,
                    "model_result": decoded_output
                }
                part_result.update(model_output)

                task_results.append(part_result)

                y_pred_sample.append(model_output['model_answer'])

                # 8. Call interpretability attention score method
                if interpr:
                    interpretability1.cal_attn(part_id=part_id, question=formatted_part, reason=part_result["model_reasoning"], answer=part_result["model_answer"], msg = chat.get_message())
                    
                    # Put model back into training mode
                    self.model.train()


            # 9. Report the results for the sample: answers and accuracy
            self.print_sample_predictions(trues=y_true_sample, preds=y_pred_sample)

            strict_sample, soft_sample = self.calculate_and_print_accuracies(
                trues_preds=(y_true_sample, y_pred_sample),
                id_=sample_id,
                what="sample",
            )
            sample_accuracies.append(strict_sample)
            sample_soft_match_accuracies.append(soft_sample)

        # 10. Report the results for the task: accuracy
        print("\n- TASK RESULTS -", end="\n\n")

        strict_task, soft_task = self.calculate_and_print_accuracies(
            str_acc_soft_acc=(sample_accuracies, sample_soft_match_accuracies),
            id_=task_id,
            what="task",
            to_print=True,
        )

        task_results[0]["accuracy"] = strict_task
        task_results[0]["soft_match_accuracy"] = soft_task
        self.accuracies_per_task.append(strict_task)
        self.soft_match_accuracies_per_task.append(soft_task)

        print(f"The work on task {task_id} is finished successfully")

        # Clear the cache
        torch.cuda.empty_cache()

        return task_results

    def get_mean_accuracies(self) -> [float, float]:
        """
        Calculate the mean accuracies for the tasks, insert them into the list of accuracies on the first position.

        :return: mean strict accuracy, mean soft match accuracy
        """
        mean_strict_accuracy = round(statistics.mean(self.accuracies_per_task), 2)
        mean_soft_match_accuracy = round(
            statistics.mean(self.soft_match_accuracies_per_task), 2
        )
        self.accuracies_per_task.insert(0, mean_strict_accuracy)
        self.soft_match_accuracies_per_task.insert(0, mean_soft_match_accuracy)
        return mean_strict_accuracy, mean_soft_match_accuracy
