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
        saver: DataSaver = None,
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

        self.saver: DataSaver = saver
        self.chat: Chat = None
        self.current_part: SamplePart = None

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
    def apply_setting(self, decoded_output: str, chat: Chat = None) -> tuple[str, int]:
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
                    f"TASK {task_id}/{self.total_tasks if self.total_tasks else 20} | "
                    f"SAMPLE {sample_id}/{len(task_data)} | "
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
                self.current_part = SamplePart(
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

                chat.add_message(part=self.current_part, source=Source.user)

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
                with torch.no_grad():
                    decoded_output = self.model.call(prompt=formatted_prompt)
                    torch.cuda.empty_cache()
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
                torch.cuda.empty_cache()

                # 6. Add the model's output to conversation
                chat.add_message(
                    part=decoded_output,
                    source=Source.assistant,
                )

                if self.multi_system:
                    with torch.no_grad():
                        # 8. Call interpretability attention score method
                        try:
                            interpretability_before = (
                                self.interpretability.get_attention(
                                    self.current_part, chat=chat, after=False
                                )
                                if self.multi_system and self.interpretability
                                else None
                            )
                        except torch.OutOfMemoryError:
                            warnings.warn(
                                "DEBUG: Out of memory error while calculating interpretability scores * before *. "
                                "Skipping this step."
                            )
                            interpretability_before = None

                        answer, reasoning = parse_output(output=decoded_output)
                        self.current_part.set_output(
                            model_output=decoded_output,
                            model_answer=answer,
                            model_reasoning=reasoning,
                            interpretability=interpretability_before,
                            version="before",
                        )

                with torch.no_grad():
                    decoded_output, feedback_iterations = self.apply_setting(
                        decoded_output=decoded_output, chat=chat
                    )

                    # Now parse the string output
                    answer, reasoning = parse_output(output=decoded_output)
                    try:
                        # 8. Call interpretability attention score method
                        interpretability_after = (
                            self.interpretability.get_attention(
                                self.current_part, chat=chat, after=True
                            )
                            if self.interpretability
                            else None
                        )
                    except torch.OutOfMemoryError:
                        warnings.warn(
                            "DEBUG: Out of memory error while calculating interpretability scores * after *. "
                            "Skipping this step."
                        )
                        interpretability_after = None

                    self.current_part.set_output(
                        model_output=decoded_output,
                        model_answer=answer,
                        model_reasoning=reasoning,
                        interpretability=interpretability_after,
                        version="after",
                        feedback_iterations=feedback_iterations,
                    )

                if self.saver:
                    result = self.current_part.get_result()
                    self.saver.save_output(
                        data=[result],
                        headers=list(result.keys()),
                        file_name=f"t_{task_id}_s_{sample_id}_results.csv",
                    )
                    if self.interpretability:
                        self.saver.save_part_interpretability(
                            self.current_part, multi_system=self.multi_system
                        )

                if self.multi_system:
                    print(
                        f"Last chat message from student after applying the setting: {chat.messages['student'][-1]}"
                    )

                sample.add_part(self.current_part)
                torch.cuda.empty_cache()
                gc.collect()

            # 9. Initially evaluate the result for a sample and aggregate results
            sample.print_sample_predictions()
            sample.calculate_metrics()
            print_metrics(sample, table=True)
            task.add_sample(sample)

            # 9.5. Save the sample result
            # if self.saver:
            #     sample.set_results()
            #     self.saver.save_sample_result(
            #         task_id=task_id,
            #         sample_id=sample_id,
            #         sample_data=sample,
            #         multi_system=self.multi_system,
            #     )

        # 10. Report the results for the task and aggregate results
        print("\n- TASK RESULTS -", end="\n\n")
        print_metrics(task, table=True)
        task.set_results()

        print(f"The work on task {task_id} is finished successfully")

        # Clear the cache
        torch.cuda.empty_cache()
        gc.collect()

        return task
