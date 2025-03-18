from __future__ import annotations

import gc
from abc import ABC, abstractmethod

import torch

from evaluation.Evaluator import AnswerEvaluator, MetricEvaluator
from inference.Chat import Chat, Source
from inference.DataLevels import SamplePart, Task, Sample
from inference.Prompt import Prompt
from interpretability.Interpretability import Interpretability
from settings.Model import Model
from settings.config import Enumerate, Wrapper


class Setting(ABC):
    """
    Abstract class for settings
    """

    @abstractmethod
    def __init__(
        self,
        model: Model,
        to_enumerate: Enumerate,
        total_tasks: int,
        total_parts: int,
        samples_per_task: int = 5,
        init_prompt: Prompt = None,
        multi_system: bool = False,
        wrapper: Wrapper = None,
        interpretability: Interpretability = None,
    ):
        """
         The setting class is an abstract class for all settings.

        :param init_prompt: system prompt to start conversations
        :param samples_per_task: number of samples per task for logging
        """
        self.model = model

        self.init_prompt = init_prompt
        self.to_enumerate = to_enumerate

        self.question_id = 0
        self.total_tasks = total_tasks
        self.total_parts = total_parts
        self.samples_per_task = samples_per_task
        self.wrapper = wrapper

        self.multi_system = multi_system

        self.interpretability = interpretability

    @abstractmethod
    def prepare_prompt(self, chat: Chat, resume_gen=False, model_role=None) -> str:
        """
        Prepares the prompt to include the current part of the sample.

        :param model_role: role of the model in the conversation
        :param resume_gen: whether to resume the generation
        :param chat: the current chat
        :return: prompt with the task and the current part
        """
        raise NotImplementedError

    @abstractmethod
    def apply_setting(self, decoded_output: str, chat: Chat = None) -> tuple:
        """
        Apply setting-specific postprocessing of the inital model output.
        For the baseline and skyline, this consists of parsing the output.
        For the SD and feedback setting, this entails the main idea of these settings.

        :param decoded_output: the current output of the student
        :param chat: the current chat, only necessary in the SD and feedback setting
        :return: parsed output
        """
        # ALSO INCLUDES SETTINGS -> SD AND FEEDBACK
        raise NotImplementedError

    def iterate_task(
        self,
        task_id: int,
        task_data: dict[int, list[dict[str, dict]]],
        prompt_name: str,
    ) -> Task:
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
        :return: results for the task in a list of dicts with each dict representing
                 one call to the model and will end up as one row of the table
        """
        task_evaluator = MetricEvaluator(level="task")
        task = Task(task_id=task_id, evaluator=task_evaluator)

        # 1. Iterate through samples
        for sample_id_, sample_parts in task_data.items():
            sample_id = sample_id_ + 1
            part_id = 0

            sample_eval = AnswerEvaluator(level="sample")
            sample_eval.golden_answers = [
                " ".join(list(part["answer"].values())[0]) for part in sample_parts
            ]
            # TODO: add silver reasoning
            sample_eval.silver_reasonings = []
            sample = Sample(task_id=task_id, sample_id=sample_id, evaluator=sample_eval)

            # each sample is a new conversation
            chat = Chat(system_prompt=self.init_prompt, multi_system=self.multi_system)

            # 2. Iterate through parts (one question at a time)
            for sample_part, y_true in zip(sample_parts, sample_eval.golden_answers):
                self.question_id += 1
                part_id += 1
                print(
                    "\n-* "
                    f"TASK {task_id}/{self.total_tasks} | "
                    f"SAMPLE {sample_id}/{self.samples_per_task if self.samples_per_task < 500 else len(task_data)} | "
                    f"PART {part_id}/{len(sample_parts)} | "
                    f"{prompt_name} | "
                    f"RUN ID {self.question_id}/{self.total_parts} "
                    "*-",
                    end="\n\n\n",
                )

                print(
                    (
                        f" ---- Student ---- "
                        if self.multi_system
                        else " ---- Model ---- "
                    ),
                    end="\n\n\n",
                    flush=True,
                )
                current_part = SamplePart(
                    id_=self.question_id,
                    task_id=task_id,
                    sample_id=sample_id,
                    part_id=part_id,
                    raw=sample_part,
                    golden_answer=y_true,
                    wrapper=self.wrapper,
                    to_enumerate=self.to_enumerate,
                )
                self.model.curr_sample_part = current_part

                print(
                    (
                        f"Formatted student prompt:"
                        if self.multi_system
                        else f"Formatted model prompt:"
                    ),
                    current_part.task,
                    sep="\n",
                    end="\n",
                )

                chat.add_message(part=current_part, source=Source.user)

                formatted_prompt = self.prepare_prompt(chat=chat)

                # 5. Call the model and yield the response
                decoded_output = self.model.call(prompt=formatted_prompt)
                print(
                    (
                        f"Formatted student prompt: \n"
                        if self.multi_system
                        else f"Formatted model prompt: \n"
                    ),
                    decoded_output,
                    "\n ------------- ",
                    end="\n\n\n",
                    flush=True,
                )

                # 6. Add the model's output to conversation
                chat.add_message(
                    part=decoded_output,
                    source=Source.assistant,
                )

                with torch.no_grad():
                    answer, reasoning = self.apply_setting(
                        decoded_output=decoded_output
                    )

                current_part.set_output(decoded_output, answer, reasoning)
                sample.add_part(current_part)

                # 7. Call interpretability attention score method
                if self.interpretability:
                    current_part.interpretability = self.interpretability.get_attention(
                        current_part, chat=chat
                    )

            sample.print_sample_predictions()

            exact_match_acc, soft_match_acc = sample_eval.calculate_accuracies()
            sample_eval.print_accuracies(
                id_=sample_id,
                exact_match_acc=exact_match_acc,
                soft_match_acc=soft_match_acc,
            )

            task.add_sample(sample)

        # 10. Report the results for the task: accuracy
        print("\n- TASK RESULTS -", end="\n\n")
        task_evaluator.print_accuracies(id_=task_id)
        task.set_results()

        print(f"The work on task {task_id} is finished successfully")

        # Clear the cache
        torch.cuda.empty_cache()
        gc.collect()

        return task
