from dataclasses import dataclass
from pathlib import Path

from data.TaskExamples import Task, TaskExample, TaskExamples
from inference.DataLevels import SamplePart
from settings.config import Examples


@dataclass
class PromptType:
    """
    This class handles the types of prompts.
    """

    init = "init"
    eval = "eval"
    resume = "resume"


class Prompt:
    """
    This class handles prompts.
    """

    def __init__(
        self,
        prompt: str = None,
        prompt_path: str = None,
        prompt_type: PromptType = "init",
        wrapper: str = None,
        name: str = None,
    ):
        """
        Create a prompt.
        A prompt can be added by either providing the prompt type which will then be read from the default prompt file,
        by providing the prompt as a string itself or by providing the path to a text file containing the prompt.

        :param prompt: str, the text of the prompt to use.
                       If no prompt is given, the default prompt according to the type is used.
        :param prompt_path: str, the path to the prompt file as an alternative to the prompt to read the prompt from
        a file.
        :param prompt_type: the type of the prompt. This can be "init", "eval" or "resume".
                            Can be used instead of prompt or prompt_path to load the default prompt.
        :param wrapper: str, the wrapper for the prompt that we want to pass with each call to the model
        :param name: str, the name of the prompt
        """
        if not (prompt or prompt_path or prompt_type):
            raise ValueError("Please provide a prompt, a prompt path or a prompt type.")

        # if no prompt is given, use default prompt
        if prompt:
            self.text: str = prompt
        elif prompt_type and not prompt_path:
            try:
                self.text: str = self.read_prompt_from_file(
                    f"./{prompt_type}_prompt.txt"
                )
            except FileNotFoundError:
                print("No prompt file found. Please create a prompt file.")
        elif prompt_path:
            self.text: str = self.read_prompt_from_file(prompt_path)
        else:
            self.text: str = "At some point, here will be a default prompt."

        self.original_text: str = self.text
        self.wrapper: str = wrapper
        self.name: str = name

    def format_teacher_sys(
        self, student_sys: str, student_chat: list[dict[str, str]]
    ) -> str:
        """
        Format the teacher's system prompt by inserting the student's init prompt and the parts of the sample the
        student has solved so far.
        This prompt is only applicable per part (NOT per sample).

        :param student_sys: str, the student's system prompt
        :param student_chat: list[dict], the chat of the student

        :return: str, the formatted teacher's system prompt
        """
        parts_so_far = ""
        parts_set = set()
        for message in student_chat:
            if message["role"] == "user" and message["content"] not in parts_set:
                parts_so_far += message["content"] + "\n"
                parts_set.add(message["content"] + "\n")

        return self.text.format(init_prompt=student_sys, parts_so_far=parts_so_far)

    def format_teacher_message(self, student_out: str):
        """
        Add the students current output into the instruction for the teacher.

        :param student_out: str, the current output of the student
        :return: str, the instruction with the student output inserted
        """
        if student_out:
            wrapped_out = self.wrapper.format(student_output=student_out)
        else:
            wrapped_out = self.wrapper.format(student_output=" ")

        return wrapped_out

    def format_resume_message(self, corrected_student_output: str) -> str:
        """
        Formulate the resume prompt for the student model.

        Use the general resume prompt and insert the current chain of thought to be resumed.
        This prompt is then used to prompt the student model to resume the chain of thought with the corrections by
        the teacher.

        :param corrected_student_output: the correct part of the student's previous output with the teacher's
        suggestion added

        :return: the formulated resume prompt
        """
        if corrected_student_output:
            wrapped_out = self.wrapper.format(to_continue=corrected_student_output)
        else:
            wrapped_out = self.wrapper.format(to_continue=" ")

        return wrapped_out

    def format_refine_message(
        self,
        model_output: str,
        teacher_feedback: str = "",
    ) -> str:
        """
        Format a refine message for the student model.

        :param model_output: the chain of thought to continue
        :param teacher_feedback: the teacher's feedback to add to the message
        :return: the formatted refine message
        """
        text = self.original_text
        return text.format(
            student_output=model_output, teacher_feedback=teacher_feedback
        )

    def format_feedback_message(self, part: SamplePart, cot_to_evaluate: str) -> str:
        """
        Format a feedback message for the teacher model.

        :param part: the part of the sample to evaluate
        :param cot_to_evaluate: the chain of thought to evaluate
        :return: the formatted feedback message
        """
        formatted_part = f"{part.structured_context}\n{part.structured_question}"

        formatted_with_cot = f"{self.text}\n{formatted_part}"
        return formatted_with_cot + "\n" + cot_to_evaluate

    @staticmethod
    def read_prompt_from_file(file_path: str):
        """
        Read the prompt from a file.

        :param file_path: the path to the file that contains the prompt
        """
        with open(Path(file_path), "r", encoding="UTF-8") as file:
            text = file.read().strip()
        return text

    @staticmethod
    def format_example(example: Task, number: int, wrapper: str) -> str:
        """
        Format the example with the wrapper.

        :param example: the example that should be wrapped
        :param number: the number of the example, if 0, no enumeration is added
        :param wrapper: the wrapper for the example

        :return: the formatted example
        """
        # if there is only one examples, no enumeration is needed
        number = f" {number}" if number > 1 else ""
        wrapped_example = wrapper.format(number=number, example=example)

        return f"\n\n{wrapped_example}"

    def use_original_prompt(self):
        """
        Use the original prompt.
        """
        self.text = self.original_text

    def add_examples(self, task_id: int, example_config: Examples) -> None:
        """
        Adds one or multiple examples to the prompt under the hood
        but also returns the prompt with the examples.

        :param task_id: the task id
        :param example_config: the example configuration with
                        - the number of examples,
                        - if they are handpicked,
                        - if they should be enumerated,
                        - the example wrapper
        """
        self.use_original_prompt()

        if example_config.number == 1:
            example = TaskExample(
                number=task_id,
                to_enumerate=example_config.enumerated,
                handpicked=example_config.handpicked,
                not_mentioned=example_config.not_mentioned,
            )
            self.text += self.format_example(
                example=example, number=0, wrapper=example_config.wrapper
            )
        else:
            if example_config.handpicked:
                raise NotImplementedError(
                    "Getting multiple handpicked examples is not implemented, use TaskExample class."
                )
            counter = 1
            for example in TaskExamples(
                number=task_id,
                to_enumerate=example_config.enumerated,
                handpicked=example_config.handpicked,
                not_mentioned=example_config.not_mentioned,
            ):
                self.text += self.format_example(
                    example=example, number=counter, wrapper=example_config.wrapper
                )
                counter += 1
                if counter > example_config.number:
                    break
