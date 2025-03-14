from __future__ import annotations

from abc import ABC, abstractmethod
import os

import torch

from evaluation.Evaluator import AnswerEvaluator, MetricEvaluator
from prompts.Chat import Chat, Source
from prompts.Prompt import Prompt
from settings.Model import Model
from settings.config import Enumerate
from interpretability.Interpretability import Interpretability



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
        total_tasks: int,
        total_parts: int,
        samples_per_task: int = 5,
        prompt: Prompt = None,
        interpretability: Interpretability = None
    ):
        """
        Baseline class manages model runs and data flows around it.
        It is intended to use in the further settings.

        :param parse_output: if we want to parse the output of the model (currently looks for 'answer' and 'reasoning')
        :param prompt: system prompt to start conversations
        :param samples_per_task: number of samples per task for logging
        """
        self.model = model

        self.prompt = prompt
        self.to_enumerate = to_enumerate
        self.parse_output = parse_output

        self.question_id = 0
        self.total_tasks = total_tasks
        self.total_parts = total_parts
        self.samples_per_task = samples_per_task

        self.interpretability = interpretability


    @abstractmethod
    def prepare_prompt(self, chat: Chat) -> str:
        """
        Prepares the prompt to include the current part of the sample.
        :param chat: the chat object
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

    def iterate_task(
        self,
        task_id: int,
        task_data: dict[int, list[dict[str, dict]]],
        prompt_name: str,
        task_evaluator: MetricEvaluator,
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
        7. parse output
        8. call interpretability attention score method
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
        :param task_evaluator: the evaluator for the task
        :return: results for the task in a list of dicts with each dict representing
                 one call to the model and will end up as one row of the table
        """
        task_results = []
        sample_eval = AnswerEvaluator(level="sample")

        # 1. Iterate through samples
        for sample_id, sample_parts in list(task_data.items()):
            # each sample is a new conversation
            chat = Chat(system_prompt=self.prompt.text)
            # collect the true answers
            sample_eval.true_values = [
                " ".join(list(part["answer"].values())[0]) for part in sample_parts
            ]
            sample_eval.predicted_values = []
            sample_id_ = sample_id + 1
            part_id = 0

            # 2. Iterate through parts (one question at a time)
            for sample_part, y_true in zip(sample_parts, sample_eval.true_values):
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

                formatted_prompt = self.prepare_prompt(chat=chat)

                # 5. Call the model and yield the response
                decoded_output = self.model.call(prompt=formatted_prompt)

                print(
                    "Model's output:",
                    decoded_output.lower(),
                    end="\n\n\n",
                    flush=True,
                )

                # 6. Add the model's output to conversation
                chat.add_message(part=decoded_output, source=Source.assistant)

                # 7. Parse output
                model_output = self.apply_setting(decoded_output=decoded_output)

                part_result = {
                    "part_id": part_id,
                    "id": self.question_id,
                    "task_id": task_id,
                    "sample_no": sample_id_,
                    "part": formatted_part.strip(),
                    "true_answer": y_true,
                }
                part_result.update(model_output)
                part_result.update(
                    sample_eval.evaluate(y_true, model_output["model_answer"])
                )

                task_results.append(part_result)

                sample_eval.predicted_values.append(part_result["model_answer"])


                # 8. Call interpretability attention score method
                if self.interpretability:
                    print("res", model_output["model_answer"], model_output["model_reasoning"])
                    interpr_scores = self.interpretability.calculate_attention(part_result, chat=chat)
                    part_result.update(interpr_scores)

            # 9. Report the results for the sample: answers and accuracy
            self.print_sample_predictions(
                trues=sample_eval.true_values, preds=sample_eval.predicted_values
            )

            exact_match_acc, soft_match_acc = sample_eval.calculate_accuracies()
            sample_eval.print_accuracies(
                id_=sample_id,
                exact_match_acc=exact_match_acc,
                soft_match_acc=soft_match_acc,
            )

        task_evaluator.update(sample_eval)

        # 10. Report the results for the task: accuracy
        print("\n- TASK RESULTS -", end="\n\n")

        task_evaluator.print_accuracies(id_=task_id)
        task_results[0][
            "exact_match_accuracy"
        ] = task_evaluator.exact_match_accuracy.get_mean()
        task_results[0][
            "soft_match_accuracy"
        ] = task_evaluator.soft_match_accuracy.get_mean()

        print(f"The work on task {task_id} is finished successfully")

        # Clear the cache
        torch.cuda.empty_cache()

        return task_results
