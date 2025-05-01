import re
from dataclasses import dataclass
from pathlib import Path

import en_core_web_sm
from transformers import PreTrainedTokenizerFast

from data.TaskExamples import Task, TaskExample, TaskExamples
from inference.utils import sents_to_ids, flatten, upd_span, Source
from settings.config import Examples

nlp = en_core_web_sm.load()


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
        history: str = None,
        name: str = None,
        tokenizer: PreTrainedTokenizerFast = None,
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
            raise ValueError(
                "Please provide a prompt string, a prompt path or a prompt type."
            )

        self.name: str = name
        self.tokenizer: PreTrainedTokenizerFast = tokenizer

        self.original_text: str = self.text
        self.examples = ""
        if self.tokenizer:
            self.orig_ids, self.orig_sent_spans = sents_to_ids(
                nlp(self.text).sents, self.tokenizer
            )
            self.offset = len(flatten(self.orig_ids))
            self.ids, self.sent_spans = self.orig_ids, self.orig_sent_spans
            self.ex_ids = []
            self.ex_sent_spans = []

        self.history = history
        self.wrapper: str = wrapper

    def add_history(self, student_messages: list[dict]):
        """
        Format the history of the teacher's system prompt by inserting the student's init prompt and
        the parts of the sample the student has solved so far.
        This prompt is only applicable per part (NOT per sample).

        :param student_messages: the messages of the student chat

        :return: str, the formatted teacher's system prompt
        """
        parts_so_far = ""
        parts_set = set()
        ids_so_far, spans_types_so_far = [], {}

        # TODO: do we need to filter out generation tokens from the student history?

        for message in student_messages:
            if message["role"] != "assistant":
                if message["content"] in parts_set:
                    raise ValueError(
                        f"Duplicate message in the chat: {message['content']}"
                    )

                parts_so_far += message["content"] + "\n\n"
                parts_set.add(message["content"] + "\n\n")

                ids_so_far.extend(message["ids"])
                spans_types_so_far.update(message["spans_types"])

        self.text = (
            self.original_text + "\n\n" + self.history.format(chat_history=parts_so_far)
        )
        self.ids = self.orig_ids + ids_so_far
        self.sent_spans = {
            **{span: "teacher sys" for span in self.orig_sent_spans},
            **spans_types_so_far,
        }

    def format_teacher_message(self, student_message: dict):
        """
        Format the teacher message by inserting the student's output into the instruction,
        both the content and the ids.

        :param student_message: the message of the student chat with the content and ids
        :return: str, the instruction with the student output inserted
        """
        teacher_string = ""
        teacher_ids = []
        for line in self.wrapper.split("\n"):
            if "student_output" in line:
                teacher_string += student_message["content"]
                teacher_ids.extend(student_message["ids"])
            else:
                teacher_string += line + "\n"
                # TODO: possibly not a good idea, maybe better tokenize, too
                teacher_ids.extend(
                    self.tokenizer.encode(line + "\n", add_special_tokens=False)
                )
        print(
            "Teacher's message:",
            teacher_string,
            sep="\n",
            end="\n\n\n",
        )
        return {
            "part": teacher_string,
            "source": Source.user,
            "ids": teacher_ids,
        }

    def format_resume_message(
        self, corrected_student_str: str, corrected_student_ids: list[int]
    ) -> str:
        """
        Formulate the resume prompt for the student model.

        Use the general resume prompt and insert the current chain of thought to be resumed.
        This prompt is then used to prompt the student model to resume the chain of thought with the corrections by
        the teacher.

        :param corrected_student_str: the correct part of the student's previous output with the teacher's
        :param corrected_stud_ids: the corresponding ids
        suggestion added

        :return: the formulated resume prompt
        """
        resume_str = ""
        resume_ids = []
        for line in self.wrapper.split("\n"):
            if "to_continue" in line:
                resume_str += corrected_student_str
                resume_ids.extend(corrected_student_ids)
            else:
                resume_str += line + "\n"
                resume_ids.extend(
                    self.tokenizer.encode(line + "\n", add_special_tokens=False)
                )
        print(
            "Formatted refine message:",
            resume_str,
            sep="\n",
            end="\n\n\n",
        )
        return {
            "part": resume_str,
            "source": Source.assistant,
            "ids": resume_ids,
        }

    # def format_feedback_message(self, part: SamplePart, cot_to_evaluate: str) -> str:
    #     """
    #     Format a feedback message for the teacher model.
    #
    #     :param part: the part of the sample to evaluate
    #     :param cot_to_evaluate: the chain of thought to evaluate
    #     :return: the formatted feedback message
    #     """
    #     formatted_part = f"{part.structured_context}\n{part.structured_question}"
    #
    #     formatted_with_cot = f"{self.text}\n{formatted_part}"
    #     return formatted_with_cot + "\n" + cot_to_evaluate

    def format_refine_message(
        self, student_message: dict, teacher_feedback: str
    ) -> dict:
        """
        Format the refine message by inserting the student's output into the instruction,
        both the content and the ids.

        :param student_message: the message of the student chat with the content and ids
        :param teacher_feedback: the feedback from the teacher
        :return: dict, the refine message for the student with content, source, and ids.
        """
        refine_str = ""
        refine_ids = []
        for line in self.original_text.split("\n"):
            if "student_output" in line:
                refine_str += student_message["content"]
                refine_ids.extend(student_message["ids"])
            elif "teacher_feedback" in line:
                refine_str += teacher_feedback
                refine_ids.extend(
                    self.tokenizer.encode(teacher_feedback, add_special_tokens=False)
                )
            else:
                refine_str += line + "\n"
                refine_ids.extend(
                    self.tokenizer.encode(line + "\n", add_special_tokens=False)
                )
        print(
            "Formatted refine message:",
            refine_str,
            sep="\n",
            end="\n\n\n",
        )
        return {
            "part": refine_str,
            "source": Source.user,
            "ids": refine_ids,
        }

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

    def process_example(self, formatted_example: str):
        """
        Process the examples by converting them to ids and spans in parts (not sentences!) for the sake of conciseness

        :param formatted_example: the formatted example
        :return: None
        """
        self.examples += formatted_example
        # process examples making bigger chunks out of the examples
        ids, spans = sents_to_ids(
            re.split(r"\n{2,}", formatted_example), self.tokenizer
        )
        self.ex_ids.extend(ids)
        self.ex_sent_spans.extend([upd_span(span, self.offset) for span in spans])

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
        self.ex_ids, self.ex_sent_spans = [], []

        if example_config.number == 1:
            example = TaskExample(
                number=task_id,
                to_enumerate=example_config.enumerated,
                handpicked=example_config.handpicked,
                not_mentioned=example_config.not_mentioned,
            )
            formatted_example = self.format_example(
                example=example, number=0, wrapper=example_config.wrapper
            )
            self.process_example(formatted_example)
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
                formatted_example = self.format_example(
                    example=example.strip(),
                    number=counter,
                    wrapper=example_config.wrapper,
                )
                self.process_example(formatted_example)
                counter += 1
                if counter > example_config.number:
                    break

        self.text += self.examples
        self.ids = self.orig_ids + self.ex_ids
        self.sent_spans = self.orig_sent_spans + self.ex_sent_spans
