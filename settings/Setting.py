from __future__ import annotations

import copy
import gc
import warnings
from abc import ABC, abstractmethod

import torch

from data.DataSaver import DataSaver
from inference.Chat import Chat, Source
from inference.DataLevels import Sample, SamplePart, Task, print_metrics
from inference.Prompt import Prompt
from interpretability.Interpretability import Interpretability
from interpretability.utils import InterpretabilityResult
from settings.Model import Model
from settings.utils import parse_output


class Setting(ABC):
    """
    Abstract class for settings
    """

    @abstractmethod
    def __init__(
        self,
        model: Model,
        total_tasks: int,
        total_parts: int,
        samples_per_task: int = 5,
        init_prompt: Prompt = None,
        multi_system: bool = False,
        interpretability: Interpretability = None,
        saver: DataSaver = None,
    ):
        """
         The setting class is an abstract class for all settings.

        :param init_prompt: system prompt to start conversations
        :param samples_per_task: number of samples per task for logging
        :param total_tasks: total number of tasks
        :param total_parts: total number of parts
        :param model: model to use
        :param multi_system: whether the chat for one sample consists of multiple systems, i.e. a teacher and a student
        :param interpretability: interpretability class to use
        :param saver: data saver to use
        """
        self.model: Model = model

        self.init_prompt: Prompt = init_prompt

        self.total_tasks: int = total_tasks
        self.total_parts: int = total_parts
        self.samples_per_task: int = samples_per_task

        self.part: SamplePart = None
        self.chat: Chat = None

        self.multi_system: bool = multi_system
        self.interpretability: Interpretability = interpretability

        self.saver: DataSaver = saver
        self.chat: Chat = None

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

    def create_chat_copy(self, chat: Chat) -> Chat:
        """
        Store the original chat and create a copy for use in the setting.

        :param chat: The original chat to copy
        :return: A copy of the original chat
        """
        self.chat = chat
        return copy.deepcopy(chat)

    @staticmethod
    def set_teacher_system_prompt(chat: Chat, teacher_sys: Prompt) -> None:
        """
        Set the system prompt for the teacher.
        This includes clearing the teacher's chat of previous parts.

        :param: chat: Chat, the current chat for the sample
        :param: teacher_sys: Prompt, the system prompt for the teacher
        :return: None
        """
        # clear the teacher's chat
        if chat.messages["teacher"]:
            chat.messages["teacher"] = []

        teacher_sys_prompt = teacher_sys.format_teacher_sys(
            student_sys=chat.messages["student"][0]["content"],
            student_chat=chat.messages["student"],
        )
        chat.messages["teacher"].append(
            {"role": Source.system, "content": teacher_sys_prompt}
        )

        print(
            f"Teacher's system prompt:",
            teacher_sys_prompt,
            sep="\n",
            end="\n\n",
            flush=True,
        )

    @abstractmethod
    def apply_setting(
        self, decoded_output: str, chat: Chat = None
    ) -> tuple[str, int, InterpretabilityResult]:
        """
        Apply setting-specific postprocessing of the initial model output.
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
        task_data: dict[int, list[SamplePart]],
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
        5. call the model (optionally with Interpretability) and yield the response
        6. applying the changes that are specific to each setting
        7. initially evaluate the result for a sample and aggregate results
        8. report the results for the task and aggregate results

        :param task_id: task id corresponding to task name in original dataset
        :param task_data: task data as a dict of task ids and samples,
                          which themselves are dicts of sample ids and lists of SampleParts
        :param prompt_name: name of the prompt
        :return: results for the task in a list of dicts with each dict representing
                 one call to the model and will end up as one row of the table
        """
        task = Task(task_id=task_id, multi_system=self.multi_system)
        for sample_id, sample_parts in task_data.items():
            # each sample is a new conversation
            chat = Chat(system_prompt=self.init_prompt, multi_system=self.multi_system)
            sample = Sample(
                task_id=task_id, sample_id=sample_id, multi_system=self.multi_system
            )
            # each part has one question
            for self.part in sample_parts:
                print(
                    "\n-* "
                    f"TASK {task_id}/{self.total_tasks} | "
                    f"SAMPLE {sample_id}/{len(task_data)} | "
                    f"PART {self.part.part_id}/{len(sample_parts)} | "
                    f"{prompt_name} | "
                    f"RUN ID {self.part.id_}/{self.total_parts} "
                    "*-",
                    end="\n\n\n",
                )
                sample.add_golden_answers(self.part.golden_answer)
                sample.add_silver_reasoning(self.part.silver_reasoning)

                chat.add_message(part=self.part, source=Source.user)
                # Only run the model if the results are not loaded
                if not self.part.result_before.model_answer:
                    formatted_prompt = self.prepare_prompt(chat=chat)
                    print(
                        f"Formatted {'STUDENT' if self.multi_system else 'MODEL'} prompt:",
                        formatted_prompt,
                        sep="\n",
                        end="\n",
                    )
                    try:
                        decoded_output, interpretability = self.model.call(
                            part=self.part, prompt=formatted_prompt, chat=chat
                        )
                        print(
                            f"The output of the {'student' if self.multi_system else 'model'}:",
                            decoded_output,
                            end="\n\n\n",
                            sep="\n",
                            flush=True,
                        )
                    except torch.OutOfMemoryError:
                        decoded_output, iterations, interpretability = "", 0, None
                        warnings.warn(
                            "DEBUG: Out of memory error while calculating interpretability scores * before *. "
                            "Skipping this step."
                        )
                    answer, reasoning = parse_output(output=decoded_output)
                    self.part.set_output(
                        model_output=decoded_output,
                        model_answer=answer,
                        model_reasoning=reasoning,
                        interpretability=interpretability,
                        version="before",
                    )

                # Add model output to current chat
                chat.add_message(
                    part=self.part.result_before.model_output,
                    source=Source.assistant,
                )
                # applying the changes that are specific to each setting
                if self.multi_system:
                    print(
                        f"Last chat message from student before applying the setting: {chat.messages['student'][-1]}"
                    )
                    try:
                        decoded_output, iterations, interpretability = self.apply_setting(
                            decoded_output=self.part.result_before.model_output, chat=chat
                        )
                    except torch.OutOfMemoryError:
                        decoded_output, iterations, interpretability = "", 0, None
                        warnings.warn(
                            "DEBUG: Out of memory error while calculating interpretability scores * before *. "
                            "Skipping this step."
                        )
                    print(
                        f"Last chat message from student after applying the setting: {chat.messages['student'][-1]}"
                    )
                    answer, reasoning = parse_output(output=decoded_output)

                    self.part.set_output(
                        model_output=decoded_output,
                        model_answer=answer,
                        model_reasoning=reasoning,
                        interpretability=interpretability,
                        iterations=iterations,
                        version="after",
                    )

                if self.saver:
                    self.saver.save_part_result(self.part)

                sample.add_part(self.part)
                torch.cuda.empty_cache()
                gc.collect()

            sample.print_sample_predictions()
            sample.calculate_metrics()
            print_metrics(sample, table=True)
            task.add_sample(sample)

            # save the sample result
            # if self.saver:
            #     sample.set_results()
            #     self.saver.save_sample_result(
            #         sample_data=sample,
            #     )

        print("\n- TASK RESULTS -", end="\n\n")
        print_metrics(task, table=True)
        task.set_results()

        print(f"The work on task {task_id} is finished successfully")

        torch.cuda.empty_cache()
        gc.collect()

        return task
