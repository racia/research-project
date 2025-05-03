import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import en_core_web_sm
from transformers import PreTrainedTokenizerFast

from data.TaskExamples import Task, TaskExample, TaskExamples
from inference.utils import Source, flatten, sents_to_ids, upd_span
from settings.config import Examples
from settings.utils import encode_wrapper

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
            self.orig_tokens, self.orig_ids, self.orig_sent_spans = sents_to_ids(
                nlp(self.text).sents, self.tokenizer
            )
            self.offset = len(flatten(self.orig_ids))
            self.orig_offset = self.offset
            self.tokens, self.ids, self.sent_spans = (
                self.orig_tokens,
                self.orig_ids,
                self.orig_sent_spans,
            )
            self.spans_with_types = {}
            self.ex_tokens = []
            self.ex_ids = []
            self.ex_sent_spans = []

        if history:
            self.history: dict = (
                encode_wrapper(history, self.tokenizer) if wrapper else None
            )
        self.wrapper: dict = (
            encode_wrapper(wrapper, self.tokenizer) if wrapper else None
        )

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
        ids_so_far, tokens_so_far = [], []
        spans_with_types_so_far = {}
        self.offset = self.orig_offset

        # TODO: do we need to filter out generation tokens from the student history?

        for key, hist in self.history.items():
            intro, outro = hist["before"], hist.get("after", hist["before"])
            chunks = [intro, *student_messages, outro]
            for chunk in chunks:
                is_history_wrapper = "role" not in chunk
                is_student_message = "role" in chunk and chunk["role"] != "assistant"
                if is_history_wrapper or is_student_message:
                    # this is a student message
                    if is_student_message and chunk["content"] in parts_set:
                        raise ValueError(
                            f"Duplicate message in the chat: {chunk['content']}"
                        )
                    assert type(chunk) is dict
                    parts_so_far += chunk["content"] + "\n\n"
                    parts_set.add(chunk["content"] + "\n\n")
                    newline_tokens = self.tokenizer.tokenize("\n\n")
                    tokens_so_far.extend(chunk["tokens"] + newline_tokens)
                    ids_so_far.extend(
                        chunk["ids"]
                        + self.tokenizer.convert_tokens_to_ids(newline_tokens)
                    )
                    if chunk.get("spans_with_types", None):
                        spans_with_types = chunk["spans_with_types"]
                        upd_spans = {
                            (span[0] + self.offset, span[1] + self.offset): f"{type_}_"
                            for span, type_ in spans_with_types.items()
                        }
                        spans_with_types_so_far.update(upd_spans)
                    else:
                        spans = {
                            (span[0] + self.offset, span[1] + self.offset): "hist"
                            for span in chunk["sent_spans"]
                        }
                        spans_with_types_so_far.update(spans)

                    self.offset += len(flatten(chunk["ids"]))

        self.text = self.original_text + "\n\n" + parts_so_far
        self.ids = self.orig_ids + ids_so_far
        self.tokens = self.orig_tokens + tokens_so_far
        self.spans_with_types = {
            **{span: "teacher sys" for span in self.orig_sent_spans},
            **spans_with_types_so_far,
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
        teacher_tokens = []
        for key, wrap in self.wrapper.items():
            intro, outro = wrap["before"], wrap.get("after", wrap["before"])
            chunks = [intro, student_message, outro]
            for chunk in chunks:
                assert type(chunk) is dict
                teacher_string += chunk["content"]
                teacher_ids.extend(chunk["ids"])
                teacher_tokens.extend(chunk["tokens"])
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
            "tokens": teacher_tokens,
        }

    def format_resume_message(
        self, corrected_student_str: str, corrected_student_tokens: list[str]
    ) -> dict:
        """
        Formulate the resume prompt for the student model.

        Use the general resume prompt and insert the current chain of thought to be resumed.
        This prompt is then used to prompt the student model to resume the chain of thought with the corrections by
        the teacher.

        :param corrected_student_str: the correct part of the student's previous output with the teacher's
        :param corrected_student_tokens: the corresponding tokens
        suggestion added

        :return: the formulated resume prompt
        """
        resume_str = ""
        resume_ids = []
        resume_tokens = []

        message = {
            "content": corrected_student_str,
            "ids": self.tokenizer.convert_tokens_to_ids(corrected_student_tokens),
            "tokens": corrected_student_tokens,
        }

        for key, wrap in self.wrapper.items():
            intro, outro = wrap["before"], wrap.get("after", wrap["before"])
            chunks = [intro, message, outro]
            for chunk in chunks:
                assert type(chunk) is dict
                resume_str += chunk["content"]
                resume_ids.extend(chunk["ids"])
                filtered_ids = [id_ for id_ in chunk["ids"] if id_ is not None]
                if len(filtered_ids) != len(chunk["ids"]):
                    warnings.warn("Some tokens were not converted to ids.")

                resume_tokens.extend(chunk["tokens"])
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
            "tokens": resume_tokens,
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
        tokens, ids, spans = sents_to_ids(
            re.split(r"\n{2,}", formatted_example), self.tokenizer
        )
        self.ex_tokens.extend(tokens)
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
        self.ex_tokens, self.ex_ids, self.ex_sent_spans = [], [], []

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
