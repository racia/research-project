from dataclasses import dataclass
from pathlib import Path

from settings.baseline.config.baseline_config import Enumerate, Wrapper
from settings.utils import structure_part


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
        prompt_type: str = None,
        prompt: str = None,
        prompt_path: str = None,
        wrapper: dict[Wrapper, str] = None,
    ):
        """
        Create a prompt.
        A prompt can be added by either providing the prompt type which will then be read from the default prompt file,
        by providing the prompt as a string itself or by providing the path to a text file containing the prompt.

        :param prompt_type: the type of the prompt. This can be "init", "eval" or "resume"
        :param prompt: str, the prompt that should be used. If no prompt is given, the default prompt according to the
        type is used.
        :param prompt_path: str, the path to the prompt file
        :param wrapper: str, the wrapper for the prompt that we want to pass with each call to the model
        """
        # if no prompt is given, use default prompt
        if prompt:
            self.text = prompt
        elif prompt_type and not prompt_path:
            try:
                self.text = self.read_prompt_from_file(f"./{prompt_type}_prompt.txt")
            except FileNotFoundError:
                print("No prompt file found. Please create a prompt file.")
        elif prompt_path:
            self.text = self.read_prompt_from_file(prompt_path)
        else:
            self.text = "At some point, here will be a default prompt."

        self.wrapper = wrapper

    def format_part(
        self, part: dict[str, dict[int, str]], to_enumerate: dict[Enumerate, bool]
    ) -> str:
        """
        Format the prompt part with the wrapper.

        :param part: the part of the prompt that should be formatted
        :param to_enumerate: if to add line numbers to the beginning of lines

        :return: the formatted prompt part
        """
        context, question = structure_part(part, to_enumerate)

        if self.wrapper:
            wrapped_question = self.wrapper.question.format(question=question)
            # there are parts that have no context lines, only a question
            if context:
                wrapped_context = self.wrapper.context.format(context=context)
                return f"{wrapped_context}\n{wrapped_question}"
            else:
                return wrapped_question
        else:
            return f"{context}\n{question}"

    def formulate_init_prompt(self, input_str: str) -> str:
        """
        Formulate the init prompt for the teacher model.

        :param input_str: the task and the questions the model should answer

        :return: the formulated evaluation prompt
        """
        prompt = self.text.format(task=input_str)

        return prompt

    def formulate_eval_prompt(self, curr_cot: str, current_task: str) -> str:
        """
        Formulate the evaluation prompt for the teacher model.

        Use the general evaluation prompt and insert the current chain of thought to be evaluated.
        This prompt is then used to prompt the teacher model to evaluate the student model's chain of thought.

        :param curr_cot: the current CoT that should be evaluated
        :param current_task: str, the current task of the model

        :return: the formulated evaluation prompt
        """
        prompt = self.text.format(task=current_task, reasoning=curr_cot)

        return prompt

    def formulate_resume_prompt(self, curr_cot: str) -> str:
        """
        Formulate the resume prompt for the student model.

        Use the general resume prompt and insert the current chain of thought to be resumed.
        This prompt is then used to prompt the student model to resume the chain of thought with the corrections by
        the teacher.

        :param curr_cot: the current CoT that should be resumed

        :return: the formulated resume prompt
        """
        prompt = self.text.format(new_cot=curr_cot)

        return prompt

    @staticmethod
    def read_prompt_from_file(file_path: str):
        """
        Read the prompt from a file.

        :param file_path: the path to the file that contains the prompt
        """
        with open(Path(file_path), "r", encoding="UTF-8") as file:
            text = file.read().strip()
        return text
