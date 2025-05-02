from __future__ import annotations

import copy
import re
import warnings

from transformers import PreTrainedTokenizerFast

from data.DataSaver import DataSaver
from inference.Prompt import Prompt
from interpretability.utils import InterpretabilityResult
from settings.Model import Model
from settings.Setting import Setting


class Feedback(Setting):
    """
    This class handles the Feedback setting.

    The feedback setting consists of a teacher model, a student model, and a tokenizer.
    The init_prompt is the initial prompt that is used to start the chain of thought of the student model.
    The feedback_prompt is used to prompt the teacher model to evaluate the chain of thought of the student.
    The refine_prompt is used to prompt the student to refine its chain of thought with feedback from the teacher.
    """

    positive_indicators = [
        "good",
        "well done",
        "right",
        "that's right",
        "that is right",
        "correct answer",
        "well reasoned",
        "perfect",
    ]
    negative_indicators = [
        "wrong",
        "error",
        "mistake",
        "check",
        "look again",
        "review",
        "reconsider",
        "verify",
        "double-check",
        "hint",
        "not quite",
    ]

    def __init__(
        self,
        student: Model,
        teacher: Model,
        total_tasks: int,
        total_parts: int,
        init_prompt: Prompt = None,
        feedback_prompt: Prompt = None,
        refine_prompt: Prompt = None,
        samples_per_task: int = 5,
        teacher_max_new_tokens: int = 200,
        student_max_new_tokens: int = 200,
        multi_system: bool = True,
        saver: DataSaver = None,
    ):
        """
        Create a feedback setting.

        :param student: The student model
        :param teacher: The teacher model
        :param total_tasks: Total number of tasks
        :param total_parts: Total number of parts
        :param init_prompt: Initial prompt for the student
        :param feedback_prompt: Prompt for the teacher to evaluate student output
        :param refine_prompt: Prompt for the student to refine based on teacher feedback
        :param samples_per_task: Number of samples per task
        :param teacher_max_new_tokens: Maximum tokens for teacher generation
        :param student_max_new_tokens: Maximum tokens for student generation
        :param multi_system: Whether to use multiple systems
        """
        # Call parent class constructor with interpretability
        super().__init__(
            model=student,
            total_tasks=total_tasks,
            total_parts=total_parts,
            init_prompt=init_prompt,
            samples_per_task=samples_per_task,
            multi_system=multi_system,
            saver=saver,
        )

        # Additional attributes specific to Feedback
        self.teacher: Model = teacher
        self.student: Model = student
        self.tokenizer: PreTrainedTokenizerFast = student.tokenizer

        self.teacher_max_new_tokens: int = teacher_max_new_tokens
        self.student_max_new_tokens: int = student_max_new_tokens

        self.feedback_prompt: Prompt = feedback_prompt
        self.refine_prompt: Prompt = refine_prompt

        self.curr_eval_dict = {
            "iterations": 0,
            "max_supp_attn": [],
            "attn_on_target": [],
        }

    # def prepare_prompt(self, chat: Chat, resume_gen=False, model_role="student") -> str:
    #     """
    #     Prepares the prompt to include the current part of the sample.
    #
    #     :param chat: the current chat
    #     :param resume_gen: whether to resume generation from the last message
    #
    #     :return: prompt with the task and the current part
    #     """
    #     if self.model.to_continue or resume_gen:
    #         formatted_prompt = self.model.tokenizer.apply_chat_template(
    #             chat.messages, tokenize=False, continue_final_message=True
    #         )
    #     else:
    #         formatted_prompt = self.model.tokenizer.apply_chat_template(
    #             chat.messages, tokenize=False, add_generation_prompt=True
    #         )
    #
    #     return formatted_prompt

    @staticmethod
    def check_feedback(teacher_feedback: str) -> bool | None:
        """
        Determine whether the teacher's feedback is valid and whether they are satisfied
        with the student's response.

        :param teacher_feedback: The feedback provided by the teacher.
        :return: bool, True if feedback is valid and indicates satisfaction, False otherwise.
        """
        if not teacher_feedback.strip():
            warnings.warn("DEBUG: Teacher returned an empty response!")
            return None

        # Clean the feedback for better pattern matching
        cleaned_feedback = re.sub(r"[*_-]", "", teacher_feedback.lower())

        # Check for malformed feedback patterns
        invalid_patterns = ["_____", "-----"]
        if any(pattern in teacher_feedback for pattern in invalid_patterns):
            warnings.warn(f"Malformed feedback detected: {teacher_feedback}")
            return False

        # Look for explicit 'correct' anywhere in the feedback
        if re.search(r"(?im)^correct\b(?!.*\bincorrect\b)", cleaned_feedback):
            # Make sure "incorrect" doesn't appear after "correct"
            return True

        # Look for explicit 'incorrect'
        if "incorrect" in cleaned_feedback:
            return False

        # Calculate weighted score (with more weight on exact matches)
        positive_score = sum(
            (
                2
                if f" {indicator} " in f" {cleaned_feedback} "
                else 1 if indicator in cleaned_feedback else 0
            )
            for indicator in Feedback.positive_indicators
        )

        negative_score = sum(
            (
                2
                if f" {indicator} " in f" {cleaned_feedback} "
                else 1 if indicator in cleaned_feedback else 0
            )
            for indicator in Feedback.negative_indicators
        )

        if positive_score > negative_score:
            return True
        elif negative_score > 0:
            return False

        # Test for content that suggests further revision is needed
        if "?" in teacher_feedback or "consider" in cleaned_feedback:
            return False

        # Default behavior - when unsure
        return False

    def give_feedback(self, student_message: dict) -> tuple[str, bool]:
        """
        Prompt the teacher to give feedback on the current chain of thought of the student.
        Similar to verify_output in SpeculativeDecoding.

        :param student_message: The student's chain of thought to evaluate
        :return: A tuple containing the teacher's feedback and a boolean indicating whether
                 the feedback indicates satisfaction
        """
        if not student_message["content"]:
            student_message["content"] = " "

        teacher_message = self.feedback_prompt.format_teacher_message(student_message)
        self.teacher.chat.add_message(**teacher_message)

        print("Golden answer:", self.part.golden_answer)

        # The teacher message is already added to the chat, so no need to pass it (the call is on the whole chat)
        teacher_feedback, _ = self.teacher.call(from_chat=True, subfolder="teacher")

        # Validate feedback
        is_valid = self.check_feedback(teacher_feedback)

        if is_valid is None:
            return self.give_feedback(student_message)

        return teacher_feedback, is_valid

    def refine(self, teacher_feedback: str) -> tuple[str, InterpretabilityResult]:
        """
        Prompt the student to refine its chain of thought according to the feedback it received from
        the teacher. Similar to the speculative_decode method in SpeculativeDecoding.

        :param teacher_feedback: The feedback generated by the teacher
        :return: str, the refined chain of thought
        """
        refine_message = self.refine_prompt.format_refine_message(
            self.student.chat.messages[-1], teacher_feedback
        )
        self.student.chat.add_message(**refine_message)
        return self.student.call(part=self.part, from_chat=True, subfolder="iterations")

    def apply_setting(
        self, decoded_output: str
    ) -> tuple[str, dict, InterpretabilityResult]:
        """
        Run the feedback setting.
        The feedback setting consists of the following steps:
        1. The student generates a chain of thought
        2. The teacher gives feedback on this chain of thought
        3. The student takes in the feedback and refines its chain of thought
        4. Step 2 and 3 are repeated until no further feedback is provided

        :return: The refined model output as a string
        """
        original_student_chat = copy.deepcopy(self.student.chat)
        self.teacher.chat = self.create_teacher_chat(
            teacher_sys=self.feedback_prompt,
            tokenizer=self.tokenizer,
        )

        self.curr_eval_dict = {
            "iterations": 0,
            "max_supp_attn": [],
            "attn_on_target": [],
        }

        print(
            " ------------- Starting Feedback ------------- ", end="\n\n\n", flush=True
        )
        print(" ---- Feedback iteration 0 ---- ", end="\n\n\n", flush=True)

        print(" ---- Teacher ---- ", end="\n\n\n", flush=True)
        feedback, is_valid = self.give_feedback(self.student.chat.messages[-1])

        print(
            "Teacher's feedback:",
            f"is valid: {is_valid}",
            "feedback:",
            feedback,
            " ------------- ",
            end="\n\n\n",
            sep="\n",
            flush=True,
        )

        iteration = 1
        # without iterations, interpretability will stay None
        interpretability = None

        if self.saver and self.part:
            # interpretability for the "first iteration" is saved as "before"
            # otherwise, if the teacher is happy, it'd be saved as empty
            self.saver.save_feedback_iteration(
                part=self.part,
                iteration=iteration,
                student_message=decoded_output,
                teacher_message=feedback,
            )
        # Loop until teacher is satisfied with student output
        while not is_valid:
            iteration += 1
            print(
                f" ---- Feedback iteration {iteration} ---- ", end="\n\n\n", flush=True
            )

            # Maximum iterations check
            if iteration > 15:
                print("Maximum feedback iterations reached. Using last student output.")
                break

            decoded_output, interpretability = self.refine(feedback)

            self.curr_eval_dict["max_supp_attn"].append(
                interpretability.max_supp_attn if interpretability else None
            )
            self.curr_eval_dict["attn_on_target"].append(
                interpretability.attn_on_target if interpretability else None
            )

            print(
                " ---- Student ---- ",
                "Refined output of student:",
                decoded_output,
                " ------------- ",
                end="\n\n\n",
                sep="\n",
                flush=True,
            )

            print(" ---- Teacher ---- ", end="\n\n\n", flush=True)
            feedback, is_valid = self.give_feedback(self.student.chat.messages[-1])

            print(
                "Teacher's feedback:",
                f"is valid: {is_valid}",
                "feedback:",
                feedback,
                " ------------- ",
                end="\n\n\n",
                sep="\n",
                flush=True,
            )
            if self.saver and self.part:
                self.saver.save_feedback_iteration(
                    part=self.part,
                    iteration=iteration,
                    student_message=decoded_output,
                    teacher_message=feedback,
                    interpretability=interpretability,
                )

        # Update the original chat's last student message with the refined output
        last_model_message = self.student.chat.messages[-1]
        original_student_chat.remove_message(-1)
        original_student_chat.move_approved_message(last_model_message)
        self.student.chat = original_student_chat
        print("DEBUG: self.student.chat updated", self.student.chat)

        # call the interpretability with the final chat
        chat_ids = self.student.chat.convert_into_datatype("ids")
        output_tensor = self.student.model(
            chat_ids,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )
        interpretability = self.student.interpretability.process_attention(
            # output tensor includes all the previous ids + the model output
            output_tensor=output_tensor,
            # chat includes the current model output but the processing should not!
            chat=self.student.chat,
            chat_ids=chat_ids,
            part=self.part,
            keyword="after",
        )

        self.curr_eval_dict = {"iterations": iteration}

        return decoded_output, self.curr_eval_dict, interpretability
