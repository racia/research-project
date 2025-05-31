from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from mailbox import FormatError

import torch
from transformers import PreTrainedTokenizerFast

from data.DataSaver import DataSaver
from inference.Chat import Chat
from inference.DataLevels import Sample, SamplePart, Task, print_metrics
from inference.Prompt import Prompt
from interpretability.utils import InterpretabilityResult
from settings.Model import Model


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
        :param saver: data saver to use
        """
        self.model: Model = model
        self.init_prompt: Prompt = init_prompt
        self.multi_system: bool = multi_system

        self.total_tasks: int = total_tasks
        self.total_parts: int = total_parts
        self.samples_per_task: int = samples_per_task

        self.part: SamplePart = None

        self.saver: DataSaver = saver

    def create_teacher_chat(
        self,
        teacher_sys: Prompt,
        tokenizer: PreTrainedTokenizerFast,
        remove_last: bool = False,
    ) -> Chat:
        """
        Set the system prompt for the teacher.
        This includes clearing the teacher's chat of previous parts.

        :param: teacher_sys: Prompt, the system prompt for the teacher
        :return: None
        """
        messages = (
            self.model.chat.messages[:-1] if remove_last else self.model.chat.messages
        )
        teacher_sys.add_history(messages)
        chat = Chat(
            model_role="teacher",
            system_prompt=teacher_sys,
            tokenizer=tokenizer,
        )
        chat.supp_sent_spans = self.model.chat.supp_sent_spans
        if "{" in teacher_sys.text:
            raise FormatError(
                "The teacher prompt is still unformatted:\n", teacher_sys.text
            )
        return chat

    def get_after_interpretability(self) -> InterpretabilityResult:
        """
        Get the interpretability result for the student after the setting is applied.
        This is used for the multi-system setting when the final version of the chat is not available till the end.
        :return: interpretability result
        """
        chat_ids = self.model.chat.convert_into_datatype("ids", identify_target=True)
        output_tensor = self.model.model(
            chat_ids,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )
        interpretability = self.model.interpretability.process_attention(
            # output tensor includes all the previous ids + the model output
            output_tensor=output_tensor,
            chat=self.model.chat,
            chat_ids=chat_ids,
            part=self.part,
            keyword="after",
        )
        return interpretability

    @abstractmethod
    def apply_setting(
        self, decoded_output: str
    ) -> tuple[str, dict, InterpretabilityResult]:
        """
        Apply setting-specific postprocessing of the initial model output.
        For the baseline and skyline, this consists of parsing the output.
        For the SD and feedback setting, this entails the main idea of these settings.

        :param decoded_output: the output of the model
        :return: parsed output
        """
        # ONLY USED FOR SETTINGS -> SD AND FEEDBACK
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
        task = Task(task_id, self.multi_system)
        for sample_id, sample_parts in task_data.items():
            sample = Sample(task_id, sample_id, self.multi_system)
            # each sample is a new conversation
            self.model.chat = Chat(
                model_role="student" if self.multi_system else "model",
                system_prompt=self.init_prompt,
                tokenizer=self.model.tokenizer,
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

                # Only run the model if the results are not loaded
                if not self.part.results:
                    decoded_output, interpretability = self.model.call(self.part)
                    self.part.set_output(
                        messages=self.model.chat.messages[-2:],
                        interpretability=interpretability,
                        version="before",
                    )
                    print(
                        f"The output of the {'student' if self.multi_system else 'model'}:",
                        decoded_output,
                        end="\n\n\n",
                        sep="\n",
                        flush=True,
                    )

                # applying the changes that are specific to each setting
                if self.multi_system:
                    print(
                        f"Last chat message from student before applying the setting: {self.model.chat.messages[-1]}"
                    )

                    decoded_output, eval_dict, interpretability = self.apply_setting(
                        decoded_output=self.part.results[-1].model_output,
                    )
                    self.saver.save_eval_dict(
                        task_id=task_id,
                        sample_id=sample_id,
                        part_id=self.part.part_id,
                        eval_dict=eval_dict,
                        file_name="eval_dict_sd.json",
                    )
                    self.part.set_output(
                        messages=self.model.chat.messages[-2:],
                        interpretability=interpretability,
                        iterations=eval_dict["iterations"],
                        version="after",
                    )
                    print(
                        f"Last chat message from student after applying the setting: {self.model.chat.messages[-1]}"
                    )

                if self.saver:
                    print("Saving part result...")
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
