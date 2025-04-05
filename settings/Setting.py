from __future__ import annotations

import copy
import gc
from abc import ABC, abstractmethod

import torch

from inference.Chat import Chat, Source
from inference.DataLevels import Sample, SamplePart, Task
from inference.Prompt import Prompt
from interpretability.Interpretability import Interpretability
from settings.Model import Model
from settings.config import Enumerate, Wrapper
from settings.utils import parse_output


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

        self.init_prompt: Prompt = init_prompt
        self.to_enumerate: Enumerate = to_enumerate

        self.question_id: int = 0
        self.total_tasks: int = total_tasks
        self.total_parts: int = total_parts
        self.samples_per_task: int = samples_per_task
        self.wrapper: Wrapper = wrapper

        self.multi_system: bool = multi_system

        self.interpretability: Interpretability = interpretability

    @abstractmethod
    def prepare_prompt(self, chat: Chat, resume_gen=False, model_role=None) -> str:
        """
        @TODO: self.to_continue check, default: add_generation_prompt
        Prepares the prompt to include the current part of the sample.

        :param model_role: role of the model in the conversation
        :param resume_gen: whether to resume the generation
        :param chat: the current chat
        :return: prompt with the task and the current part
        """
        raise NotImplementedError

    @abstractmethod
    def apply_setting(self, decoded_output: str, chat: Chat = None) -> str:
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

    def create_chat_copy(self, chat: Chat) -> Chat:
        """
        Store the original chat and create a copy for use in the setting.

        :param chat: The original chat to copy
        :return: A copy of the original chat
        """
        self.chat = chat
        return copy.deepcopy(chat)

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
        7. applying the changes that are specific to each setting
        8. call interpretability attention score method
        9. initially evaluate the result for a sample and aggregate results
        10. report the results for the task and aggregate results

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
        task = Task(task_id=task_id, multi_system=self.multi_system)

        if not interpr.switch:
            interpr = None
        else:
            # Initialize interpretability for each new sample - since new chat
            interpretability1 = Interpretability(self.model, self.model.tokenizer, task_id)

        # 1. Iterate through samples
        for sample_id_, sample_parts in task_data.items():
            sample_id = sample_id_ + 1
            part_id = 0

            sample = Sample(
                task_id=task_id, sample_id=sample_id, multi_system=self.multi_system
            )
            golden_answers = [
                " ".join(list(part["answer"].values())[0]) for part in sample_parts
            ]
            sample.add_golden_answers(golden_answers)
            # TODO: add silver reasoning
            # each sample is a new conversation
            chat = Chat(system_prompt=self.init_prompt, multi_system=self.multi_system)

            # 2. Iterate through parts (one question at a time)
            for sample_part, golden_answer in zip(sample_parts, golden_answers):
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
                    golden_answer=golden_answer,
                    wrapper=self.wrapper,
                    to_enumerate=self.to_enumerate,
                    multi_system=self.multi_system,
                )

                chat.add_message(part=current_part, source=Source.user)

                formatted_prompt = self.prepare_prompt(chat=chat)

                print(
                    (
                        f"Formatted student prompt:"
                        if self.multi_system
                        else f"Formatted model prompt:"
                    ),
                    formatted_prompt,
                    sep="\n",
                    end="\n",
                )

                # 5. Call the model and yield the response
                decoded_output = self.model.call(prompt=formatted_prompt)
                print(
                    (
                        f"The output of the student:"
                        if self.multi_system
                        else f"The output of the model:"
                    ),
                    decoded_output,
                    "\n ------------- ",
                    end="\n\n\n",
                    sep="\n",
                    flush=True,
                )

                # 6. Add the model's output to conversation
                chat.add_message(
                    part=decoded_output,
                    source=Source.assistant,
                )

                if self.multi_system:
                    with torch.no_grad():
                        # 8. Call interpretability attention score method
                        interpretability_before = (
                            self.interpretability.get_attention(
                                current_part, chat=chat, after=False
                            )
                            if self.multi_system and self.interpretability
                            else None
                        )
                        answer, reasoning = parse_output(output=decoded_output)
                        current_part.set_output(
                            model_output=decoded_output,
                            answer=answer,
                            reasoning=reasoning,
                            interpretability=interpretability_before,
                            after=False,
                        )
                    print(
                        f"Last chat message from student before applying the setting: {chat.messages['student'][-1]}"
                    )
                    print(
                        "DEBUG: Model Output before applying the setting:",
                        decoded_output,
                    )

                if self.multi_system:
                    self.model.curr_sample_part = current_part

                # 7. Applying the changes that are specific to each setting
                with torch.no_grad():
                    decoded_output = self.apply_setting(
                        decoded_output=decoded_output, chat=chat
                    )
                    answer, reasoning = parse_output(output=decoded_output)
                    # 8. Call interpretability attention score method
                    interpretability_after = (
                        self.interpretability.get_attention(
                            current_part, chat=chat, after=True
                        )
                        if self.interpretability
                        else None
                    )
                    current_part.set_output(
                        model_output=decoded_output,
                        answer=answer,
                        reasoning=reasoning,
                        interpretability=interpretability_after,
                        after=True,
                    )

                if self.multi_system:
                    print(
                        f"Last chat message from student after applying the setting: {chat.messages['student'][-1]}"
                    )

                sample.add_part(current_part)

            sample.print_sample_predictions()

            # 9. Initially evaluate the result for a sample and aggregate results
            if self.multi_system:
                print("Before the setting was applied:")
                exact_match_acc_before, soft_match_acc_before = (
                    sample.evaluator_before.calculate_accuracies()
                )
                sample.evaluator_before.print_accuracies(
                    id_=sample_id,
                    exact_match_acc=exact_match_acc_before,
                    soft_match_acc=soft_match_acc_before,
                )
            exact_match_acc, soft_match_acc = (
                sample.evaluator_after.calculate_accuracies()
            )
            sample.evaluator_after.print_accuracies(
                id_=sample_id,
                exact_match_acc=exact_match_acc,
                soft_match_acc=soft_match_acc,
            )

            task.add_sample(sample)

        # 10. Report the results for the task and aggregate results
        print("\n- TASK RESULTS -", end="\n\n")
        if self.multi_system:
            print("Before the setting was applied:")
            task.evaluator_before.print_accuracies(id_=task_id)

        task.evaluator_after.print_accuracies(id_=task_id)
        task.set_results()

        print(f"The work on task {task_id} is finished successfully")

        # Clear the cache
        torch.cuda.empty_cache()
        gc.collect()

        return task
