import torch
import re
from inference.Chat import Chat, Source
from inference.Prompt import Prompt
from settings.Model import Model
from settings.Setting import Setting
from settings.utils import Enumerate
from settings.config import Wrapper
from interpretability.Interpretability import Interpretability



class Feedback(Setting):
    """
    This class handles the Feedback setting.

    The feedback setting consists of a teacher model, a student model, and a tokenizer.
    The init_prompt is the initial prompt that is used to start the chain of thought of the student model.
    The feedback_prompt is used to prompt the teacher model to evaluate the chain of thought of the student.
    The refine_prompt is used to prompt the student to refine its chain of thought with feedback from the teacher.
    """

    def __init__(
            self,
            student: Model,
            teacher: Model,
            to_enumerate: Enumerate,

            total_tasks: int,
            total_parts: int,
            interpretability: Interpretability = None,
            init_prompt: Prompt = None,
            feedback_prompt: Prompt = None,
            refine_prompt: Prompt = None,
            samples_per_task: int = 5,
            teacher_max_new_tokens: int = 200,
            student_max_new_tokens: int = 200,
            multi_system: bool = True,
            wrapper: Wrapper = None,
    ):
        """
        Create a feedback setting.
        """
        """
        Create a feedback setting.

        :param student: The student model
        :param teacher: The teacher model
        :param to_enumerate: Whether to enumerate the context and questions
        :param total_tasks: Total number of tasks
        :param total_parts: Total number of parts
        :param interpretability: Optional interpretability instance
        :param init_prompt: Initial prompt for the student
        :param feedback_prompt: Prompt for the teacher to evaluate student output
        :param refine_prompt: Prompt for the student to refine based on teacher feedback
        :param samples_per_task: Number of samples per task
        :param teacher_max_new_tokens: Maximum tokens for teacher generation
        :param student_max_new_tokens: Maximum tokens for student generation
        :param multi_system: Whether to use multiple systems
        :param wrapper: Wrapper for prompts
        """
        # Call parent class constructor with interpretability
        super().__init__(
            model=student,
            total_tasks=total_tasks,
            total_parts=total_parts,
            init_prompt=init_prompt,
            samples_per_task=samples_per_task,
            multi_system=multi_system,
            to_enumerate=to_enumerate,
            wrapper=wrapper,
            interpretability=interpretability,
        )

        # Additional attributes specific to Feedback
        self.teacher: Model = teacher
        self.student: Model = student
        self.tokenizer = student.tokenizer

        self.teacher_max_new_tokens = teacher_max_new_tokens
        self.student_max_new_tokens = student_max_new_tokens

        self.feedback_prompt = feedback_prompt
        self.refine_prompt = refine_prompt

        self.initial_student_output = None
        self.chat = None

    def prepare_prompt(self, chat: Chat, resume_gen=False, model_role="student") -> str:
        """
        Prepares the prompt to include the current part of the sample.

        :param chat: the current chat
        :param resume_gen: whether to resume generation from the last message
        :param model_role: the role of the model, either student or teacher

        :return: prompt with the task and the current part
        """
        if self.model.to_continue or resume_gen:
            formatted_prompt = self.model.tokenizer.apply_chat_template(
                chat.messages[model_role], tokenize=False, continue_final_message=True
            )
        else:
            formatted_prompt = self.model.tokenizer.apply_chat_template(
                chat.messages[model_role], tokenize=False, add_generation_prompt=True
            )

        return formatted_prompt

    def check_feedback(self, teacher_feedback: str) -> bool:
        """
        Determine whether the teacher's feedback is valid and whether they are satisfied
        with the student's response.

        :param teacher_feedback: The feedback provided by the teacher.
        :return: bool, True if feedback is valid and indicates satisfaction, False otherwise.
        """

        if not teacher_feedback.strip():
            print("Teacher returned an empty response!")
            return False

        # Clean the feedback for better pattern matching
        cleaned_feedback = re.sub(r'[*_-]', '', teacher_feedback.lower())

        # Check for malformed feedback patterns
        invalid_patterns = ["_____", "-----"]
        if any(pattern in teacher_feedback for pattern in invalid_patterns):
            print(f"Malformed feedback detected: {teacher_feedback}")
            return False

        # Look for explicit 'correct' anywhere in the feedback
        if re.search(r'\bcorrect\b(?!.*\bincorrect\b)', cleaned_feedback):
            # Make sure "incorrect" doesn't appear after "correct"
            return True

        # Look for explicit 'incorrect'
        if 'incorrect' in cleaned_feedback:
            return False

        # More nuanced positive/negative detection
        positive_indicators = ["good", "well done", "right", "that's right", "that is right",
                               "correct answer", "well reasoned", "perfect"]

        negative_indicators = ["wrong", "error", "mistake", "check",
                               "look again", "review", "reconsider", "verify",
                               "double-check", "hint", "not quite"]

        # Calculate weighted score (with more weight on exact matches)
        positive_score = sum(2 if f" {indicator} " in f" {cleaned_feedback} " else
                             1 if indicator in cleaned_feedback else 0
                             for indicator in positive_indicators)

        negative_score = sum(2 if f" {indicator} " in f" {cleaned_feedback} " else
                             1 if indicator in cleaned_feedback else 0
                             for indicator in negative_indicators)

        if positive_score > negative_score:
            return True
        elif negative_score > 0:
            return False

        # Test for content that suggests further revision is needed
        if "?" in teacher_feedback or "consider" in cleaned_feedback:
            return False

        # Default behavior - when unsure
        return False


    def generate_student(self, input_prompt: str, chat: Chat) -> str:
        """
        Generate some candidates using the student model.

        The student model generates a chain of thought based on the input string.
        The maximum length of this chain of thought is handled by the max_length parameter.

        :param input_prompt: the init prompt with the input string
        :param chat: the current chat

        :return: The output of the student model
        """
        formulated_resume_prompt = self.refine_prompt.formulate_refine_prompt(
            part=self.student.curr_sample,
            to_enumerate=self.to_enumerate,
            cot_to_continue=input_prompt,
        )

        chat.add_message(
            part=formulated_resume_prompt, source="user", model_role="student"
        )

        formatted_prompt = self.prepare_prompt(
            chat=chat, resume_gen=False, model_role="student"
        )

        print(
            "Formatted resume prompt:",
            formulated_resume_prompt,
            sep="\n",
            end="\n\n\n",
        )

        with torch.no_grad():
            student_out = self.student.call(
                formatted_prompt,
            )

            return student_out

    def give_feedback(self, input_prompt: str, chat: Chat) -> tuple[str, bool]:
        """
        Prompt the teacher to give feedback on the current chain of thought of the student.
        Similar to verify_output in SpeculativeDecoding.

        :param input_prompt: The student's chain of thought to evaluate
        :param chat: The current chat

        :return: A tuple containing the teacher's feedback and a boolean indicating whether
                 the feedback indicates satisfaction
        """
        self.teacher.curr_sample_part = self.model.curr_sample_part


        # Format the feedback prompt
        formatted_feedback_prompt = self.feedback_prompt.format_feedback_message(
            part=self.teacher.curr_sample_part,
            to_enumerate=self.to_enumerate,
            cot_to_evaluate=input_prompt
        )
        if self.teacher.curr_sample_part is None:
            raise ValueError("ERROR: teacher.curr_sample_part is None. Ensure it is properly assigned.")

        if not hasattr(self.teacher.curr_sample_part, "raw"):
            raise ValueError(
                f"ERROR: teacher.curr_sample_part does not have attribute 'raw'. Content: {self.teacher.curr_sample_part}")

        # Add to chat
        chat.add_message(part=formatted_feedback_prompt, source=Source.user, model_role="teacher")

        formatted_prompt = self.prepare_prompt(chat=chat, resume_gen=False, model_role="teacher")

        # Get teacher's response
        with torch.no_grad():
            teacher_feedback = self.teacher.call(formatted_prompt)

        print(
            "teacher_feedback:",
            teacher_feedback,
            sep="\n",
            end="\n\n\n",
        )

        # Validate feedback
        is_valid = self.check_feedback(teacher_feedback)

        return teacher_feedback, is_valid

    def refine(self, input_prompt, teacher_feedback: str, chat: Chat) -> str:
        """
        Prompt the student to refine its chain of thought according to the feedback it received from
        the teacher. Similar to the speculative_decode method in SpeculativeDecoding.

        :param input_prompt: The student's previous chain of thought
        :param teacher_feedback: The feedback generated by the teacher
        :param chat: The current chat

        :return: str, the refined chain of thought
        """
        formulated_refine_prompt = self.refine_prompt.format_refine_message(
            part=self.student.curr_sample_part,
            to_enumerate=self.to_enumerate,
            cot_to_continue=input_prompt,
            teacher_feedback=teacher_feedback
        )

        chat.add_message(
            part=formulated_refine_prompt, source="user", model_role="student"
        )

        formatted_prompt = self.prepare_prompt(
            chat=chat, resume_gen=False, model_role="student"
        )

        print(
            "Formatted resume prompt:",
            formulated_refine_prompt,
            sep="\n",
            end="\n\n\n",
        )

        with torch.no_grad():
            student_out = self.student.call(
                formatted_prompt,
            )

            return student_out

    def set_teacher_system_prompt(self, chat: Chat):
        """
        Set the system prompt for the teacher.
        This includes clearing the teacher's chat of previous parts, similar to SD.

        :param chat: Chat, the current chat for the sample
        """
        # Clear the teacher's chat
        if chat.messages["teacher"]:
            chat.messages["teacher"] = []

        teacher_sys_prompt = self.feedback_prompt.format_teacher_sys(
            student_sys=chat.messages["student"][0]["content"],
            student_chat=chat.messages["student"],
        )
        chat.messages["teacher"].append(
            {"role": Source.system, "content": teacher_sys_prompt}
        )

        print(
            "\n--------------\n",
            f"Teacher's system prompt: {teacher_sys_prompt}",
            end="\n--------------\n\n",
            flush=True,
        )

    def apply_setting(self, decoded_output: str, chat: Chat = None) -> str:
        """
        Run the feedback setting.
        The feedback setting consists of the following steps:
        1. The student generates a chain of thought
        2. The teacher gives feedback on this chain of thought
        3. The student takes in the feedback and refines its chain of thought
        4. Step 2 and 3 are repeated until no further feedback is provided

        :param decoded_output: The student's initial output
        :param chat: The current chat
        :return: The refined model output as a string
        """
        self.initial_student_output = decoded_output


        #self.student.curr_sample_part = self.model.curr_sample_part
        self.student.curr_sample_part =self.student.curr_sample_part

        self.set_teacher_system_prompt(chat=chat)
        chat = self.create_chat_copy(chat=chat)

        print(" ------------- Starting Feedback ------------- ", end="\n\n\n", flush=True)
        print(f" ---- Feedback iteration 1 ---- ", end="\n\n\n", flush=True)

        # Work with the complete student output
        student_output = decoded_output

        print(" ---- Teacher ---- ", end="\n\n\n", flush=True)
        feedback, is_valid = self.give_feedback(student_output, chat)

        print(
            "Teacher's feedback:",
            f"is valid: {is_valid}, feedback: {feedback}",
            "\n ------------- ",
            end="\n\n\n",
            flush=True,
        )

        i = 1

        # Loop until teacher is satisfied with student output
        while not is_valid:
            i += 1
            print(f" ---- Feedback iteration {i} ---- ", end="\n\n\n", flush=True)

            # Maximum iterations check
            if i > 15:
                print("Maximum feedback iterations reached. Using last student output.")
                break

            student_output = self.refine(student_output, feedback, chat)

            print(
                " ---- Student ---- \n",
                "Refined output of student:",
                student_output,
                "\n ------------- ",
                end="\n\n\n",
                flush=True,
            )

            chat.add_message(
                part=student_output, source="assistant", model_role="student"
            )

            print(" ---- Teacher ---- ", end="\n\n\n", flush=True)
            feedback, is_valid = self.give_feedback(student_output, chat)
            chat.add_message(part=feedback, source="assistant", model_role="teacher")

            print(
                "Teacher's feedback:",
                f"is valid: {is_valid}, feedback: {feedback}",
                "\n ------------- ",
                end="\n\n\n",
                flush=True,
            )

        # Update the original chat's last student message with the refined output
        if self.chat:
            self.chat.messages["student"][-1]["content"] = student_output

        # Return the final output string
        return student_output