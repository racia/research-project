from __future__ import annotations

import re

import torch

from inference.Chat import Chat, Source
from inference.Prompt import Prompt
from interpretability.Interpretability import Interpretability
from settings.Model import Model
from settings.Setting import Setting
from settings.config import Wrapper
from settings.utils import Enumerate, parse_output


class SpeculativeDecoding(Setting):
    """
    This class handles everything related to speculative decoding.
    """

    def __init__(
        self,
        student: Model,
        teacher: Model,
        to_enumerate: Enumerate,
        total_tasks: int,
        total_parts: int,
        init_prompt: Prompt = None,
        eval_prompt: Prompt = None,
        resume_prompt: Prompt = None,
        samples_per_task: int = 5,
        multi_system: bool = True,
        wrapper: Wrapper = None,
        interpretability: Interpretability = None,
    ):
        """
        Initialize the speculative decoding setting.

        The speculative decoding setting consists of a teacher model, a student model, and a tokenizer.
        The init_prompt is the initial prompt that is used to start the chain of thought of the student model.
        The resume_prompt is used to prompt the student model to resume the chain of thought with the corrections by
        the teacher.
        The eval_prompt is used to prompt the teacher to evaluate the chain of thought of the student model.

        :param teacher: The teacher model
        :param student: The student model
        :param eval_prompt: the evaluation prompt for the teacher
        :param resume_prompt: the resume prompt for the student
        """
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
        self.teacher = teacher
        self.student = student
        self.tokenizer = student.tokenizer

        self.eval_prompt = eval_prompt
        self.resume_prompt = resume_prompt

        self.initial_student_output = None

    def generate_student(self, corrected_in: str, chat: Chat) -> str:
        """
        Generate some candidates using the student model.

        The student model generates a chain of thought based on the input string.
        The maximum length of this chain of thought is handled by the max_length parameter.

        :param corrected_in: the correct part of the student's previous output with the teacher's suggestion
        :param chat: the current chat

        :return: The output of the student model
        """
        # add the evaluation prompt as a user message
        chat.add_message(
            part=self.resume_prompt.text, source="user", model_role="student"
        )
        print(
            "\n--------------\n",
            "Evaluation prompt of the user: ",
            self.resume_prompt.text,
            sep="\n",
            end="\n--------------\n\n",
        )

        # add the output the student should continue as an assistant message
        message_to_continue = self.resume_prompt.format_resume_message(
            corrected_in=corrected_in
        )
        chat.add_message(
            part=message_to_continue, source="assistant", model_role="student"
        )
        print(
            "\n--------------\n",
            "Message that the student should continue: ",
            message_to_continue,
            sep="\n",
            end="\n--------------\n\n",
        )

        formatted_prompt = self.prepare_prompt(
            chat=chat, resume_gen=True, model_role="student"
        )

        with torch.no_grad():
            student_out = self.student.call(
                formatted_prompt,
            )

            return student_out

    def verify_output(
        self, student_tokens, chat: Chat, k=10, last_err_ix=0, student_str=None
    ) -> tuple[bool, int | None, str | None]:
        """
        Verify the candidates using the teacher model.

        Let the teacher go through the whole chain of thought proposed by the student.
        If the teacher disagrees at some point, return the teachers suggestion as the CoT step.

        :param student_tokens: list of output tokens of the student model
        :param chat: the current chat
        :param k: the number of top candidates to consider from the teacher model
        :param last_err_ix: the index of the last error
        :param student_str: the output of the student model as a string

        :return: A tuple containing a boolean indicating whether the current CoT is valid, an integer or None indicating
        the error index and the teacher's intervention or None
        """
        suggested_token = None
        ix = 0

        for ix, student_token in enumerate(student_tokens):
            print(
                f"Verifying token '{student_token}' at index {ix}",
                end="\n\n",
                flush=True,
            )
            if ix > 0:
                print(
                    f"Tokens accepted by the teacher in this iteration so far: {student_tokens[:ix]}",
                    end="\n\n",
                    flush=True,
                )
            if ix <= last_err_ix:
                print(
                    f"Token '{student_tokens[:ix]}' was accepted by the teacher in a previous iteration",
                    end="\n\n",
                    flush=True,
                )
                continue

            if student_token:
                student_token = student_token.lower().strip()
            else:
                # handle empty tokens by setting it to whitespace
                student_token = " "

            # the teacher should not see the current token, as it needs to be evaluated
            if ix > 0:
                match = self.check_match(
                    token_list=student_tokens[:ix], string=student_str
                )
                if match:
                    student_out = str(match.group(0))
                else:
                    student_out = "".join(student_tokens[:ix])
            else:
                student_out = " "

            formatted_prompt = self.eval_prompt.format_teacher_message(
                student_out=student_out
            )

            print(
                "\n--------------\n",
                "Formatted teacher prompt:",
                formatted_prompt,
                sep="\n",
                end="\n--------------\n\n",
            )

            chat.add_message(
                part=formatted_prompt, source=Source.user, model_role="teacher"
            )

            formatted_eval_prompt = self.prepare_prompt(
                chat=chat, model_role="teacher", resume_gen=True
            )

            input_ids = self.tokenizer.encode(
                formatted_eval_prompt, return_tensors="pt", add_special_tokens=True
            ).to("cuda:1")

            teacher_probs = self.teacher.call_probs(input_ids=input_ids)
            top_tokens_decoded, top_tokens_encoded = self.get_top_k_tokens(
                teacher_probs=teacher_probs, k=k, skip_eos=True
            )

            suggested_token = str(top_tokens_decoded[0])

            try:
                student_token_id = self.tokenizer.encode(
                    student_token,
                    add_special_tokens=False,
                )[0]
            except IndexError:
                print(f"Student token {student_token} is out of range.")
                student_token_id = None

            print(
                "Student's token:",
                student_token,
                student_token_id,
                sep=" ",
                end="\n\n",
            )

            # check decoded top tokens contain student token AND if student_token_id is in top token ids of teacher
            if (
                student_token not in top_tokens_decoded
                and student_token_id not in top_tokens_encoded
            ):
                break

        is_eos = self.check_if_teacher_eos(chat, suggested_token)

        # student's CoT is not empty and no disagreement was found
        if is_eos and ix == len(student_tokens) - 1:
            return True, None, None

        return False, ix, suggested_token

    def get_top_k_tokens(self, teacher_probs, k, skip_eos=True) -> tuple[list, list]:
        """
        Get the top k tokens from the teacher model that have a probability higher than min_prob.

        :param teacher_probs: the probabilities of the teacher model output
        :param k: the number of top tokens to return
        :param skip_eos: boolean indicating whether to skip the EOS token

        :return: a tuple containing the top k tokens in decoded and encoded form
        """
        top_tokens_decoded = []
        top_tokens_encoded = []
        j = 0
        # some of the top tokens could be special tokens
        # we generate until we have k valid tokens
        while (
            len(top_tokens_decoded) < k and (k + j) < teacher_probs.shape[-1] and j < 20
        ):
            top_probs, top_ids = torch.topk(teacher_probs, k + j, dim=-1)

            for token_id in top_ids[0]:
                decoded_token = (
                    self.tokenizer.decode(token_id, skip_special_tokens=skip_eos)
                    .lower()
                    .strip()
                )

                if decoded_token not in top_tokens_decoded:
                    top_tokens_decoded.append(decoded_token)
                    top_tokens_encoded.append(int(token_id.item()))

                if len(top_tokens_decoded) == k:
                    break

            j += 1

        print(
            "Teacher's top tokens:",
            top_tokens_decoded,
            top_tokens_encoded,
            sep=" ",
            end="\n",
            flush=True,
        )

        return top_tokens_decoded, top_tokens_encoded

    def check_if_teacher_eos(
        self,
        chat: Chat,
        suggested_token: str,
    ) -> bool:
        """
        Check if the teacher model would generate an end-of-sentence (EOS) token next.

        :param chat: Chat, the current chat
        :param suggested_token: str, the token that the teacher suggests

        :return: boolean indicating whether the teacher model would generate an EOS token next
        """
        print(f"Checking if teacher would generate EOS next", end="\n\n\n", flush=True)
        # check if current suggested token is EOS
        if self.tokenizer.eos_token == suggested_token:
            return True

        # check if next token would be EOS
        formatted_eval_prompt = self.prepare_prompt(
            chat=chat, model_role="teacher", resume_gen=True
        )

        input_ids = self.tokenizer.encode(
            formatted_eval_prompt, return_tensors="pt", add_special_tokens=True
        ).to("cuda:1")

        probs = self.teacher.call_probs(input_ids=input_ids)
        top_tokens_decoded, top_tokens_encoded = self.get_top_k_tokens(
            teacher_probs=probs, k=5, skip_eos=False
        )
        if (
            self.tokenizer.eos_token in top_tokens_decoded
            or self.tokenizer.eos_token_id in top_tokens_encoded
        ):
            return True
        return False

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

    def get_prev_corrections(
        self,
        prev_output: str,
        student_tokens: list,
        error_id: int,
        teacher_intervention: str,
        decoded_out: str,
    ):
        """
        Get the previous corrections which should be used to prompt the student.

        :param prev_output: the current student chain
        :param student_tokens: the student tokens
        :param error_id: the error id as returned by the teacher
        :param teacher_intervention: the suggested token by the teacher
        :param decoded_out: the output of the student

        :return: the corrected chain as a list and the corrected chain as a string
        """
        prev_output = [token.strip() if token else token for token in prev_output]
        corrected_str = ""

        if prev_output:
            student_chain = prev_output + student_tokens[:error_id]
            corrected_chain = student_tokens + [teacher_intervention]
        elif error_id != 0:
            student_chain = student_tokens[:error_id]
            corrected_chain = student_tokens + [teacher_intervention]
        else:
            student_chain = []
            corrected_chain = [teacher_intervention]
            corrected_str = teacher_intervention

        if len(student_chain) >= 1:
            match = self.check_match(token_list=student_chain, string=decoded_out)

            if match:
                corrected_str = str(match.group(0))
            else:
                corrected_str = " ".join(student_chain)

            corrected_str += (
                " " + teacher_intervention
                if teacher_intervention.isalpha()
                and not (teacher_intervention in ["ing", "ed", "s"])
                else teacher_intervention
            )
        print(
            "\n--------------\n",
            "Corrected input to student: ",
            corrected_str,
            end="\n--------------\n\n",
            flush=True,
        )

        return corrected_chain, corrected_str

    def check_match(self, token_list, string):
        """
        Check if the token list matches the string.

        :param token_list: the list of tokens
        :param string: the string to check

        :return: boolean indicating whether the token list matches the string
        """
        token_list = [token.strip() if token else token for token in token_list]
        pattern = r"\s*".join(map(re.escape, token_list))
        pattern = r"\s*" + pattern

        match = re.match(pattern, string)

        return match

    def apply_setting(self, decoded_output: str, chat: Chat = None) -> tuple:
        """
        Run the speculative decoding for one instance.

        The speculative decoding consists of the following steps:
        1. Generate a chain of thought with the student model.
        2. Verify that chain of thought using the teacher model.
        3. If the teacher model disagrees, correct the chain of thought and continue with the student model.
        4. Repeat steps 2 and 3 until the teacher model agrees with the chain of thought.

        :param decoded_output: the current output of the student
        :param chat: the current chat, only necessary in the SD and feedback setting
        :return: parsed output
        """
        if not chat:
            raise ValueError("Chat is required for speculative decoding.")

        # save the initial student output as a fallback solution
        self.initial_student_output = decoded_output

        self.teacher.curr_sample_part = self.student.curr_sample_part

        self.set_teacher_system_prompt(chat=chat)

        print(
            " ------------- Starting speculative decoding ------------- ",
            end="\n\n\n",
            flush=True,
        )
        print(f" ---- SD iteration 1 ---- ", end="\n\n\n", flush=True)

        student_token_ids = self.tokenizer.encode(
            decoded_output,
            add_special_tokens=False,
        )

        student_tokens = [
            self.tokenizer.decode(to_decode, skip_special_tokens=True)
            for to_decode in student_token_ids
        ]

        is_valid, error_id, teacher_intervention = self.verify_output(
            student_tokens=student_tokens, chat=chat, student_str=decoded_output
        )

        print(
            "Teacher's evaluation:",
            f"is valid: {is_valid}, error_id: {error_id}, teacher_intervention: {teacher_intervention}",
            "\n ------------- ",
            end="\n\n",
            flush=True,
        )

        chat.add_message(
            part=teacher_intervention, source="assistant", model_role="teacher"
        )

        if not is_valid:
            decoded_output = self.speculative_decode(
                student_tokens, is_valid, error_id, teacher_intervention, chat
            )

        model_out_parsed = parse_output(decoded_output)

        return model_out_parsed

    def speculative_decode(
        self,
        student_tokens: list,
        is_valid: bool,
        error_id: int | None,
        teacher_intervention: str | None,
        chat: Chat,
    ) -> dict[str, str | None] | str:
        """
        Apply the speculative decoding on the output of the student.

        :param student_tokens: the student tokens
        :param is_valid: boolean indicating whether the current CoT is valid
        :param error_id: the error index
        :param teacher_intervention: the teacher's suggestion
        :param chat: the current chat

        :return: str, the speculative decoding output
        """
        decoded_output = self.initial_student_output

        print(
            " ---- Teacher ---- ",
            end="\n\n",
            flush=True,
        )

        # at this point, the teacher has evaluated once and potentially given a correction
        revs = 1
        approved_tokens = []
        approved_str = ""
        student_eos = False

        while (
            not is_valid and revs < 15 and not student_eos
        ):  # TODO: decide on max revisions
            print(f" ---- SD iteration {revs} ---- ", end="\n\n", flush=True)
            revs += 1
            corrected_chain, corrected_str = self.get_prev_corrections(
                approved_tokens,
                student_tokens,
                error_id,
                teacher_intervention,
                decoded_out=decoded_output,
            )

            approved_tokens += student_tokens[:error_id]

            match = self.check_match(approved_tokens, decoded_output)
            if match:
                approved_str += str(match.group(0))
            else:
                approved_str += " ".join(approved_tokens)

            # check for repetitions
            count_last_token = corrected_chain[-5:].count(corrected_chain[-1])
            if count_last_token >= 3:
                print(
                    f"Last token {corrected_chain[-1]} repeated {count_last_token} times, stopping speculative "
                    f"decoding",
                    end="\n\n\n",
                    flush=True,
                )
                break

            decoded_output = self.generate_student(
                corrected_in=corrected_str, chat=chat
            )
            print(
                " ---- Student ---- \n",
                "Resumed output of student:",
                decoded_output,
                end="\n--------------\n\n",
                flush=True,
            )
            chat.add_message(
                part=decoded_output, source="assistant", model_role="student"
            )

            student_token_ids = self.tokenizer.encode(
                decoded_output,
                add_special_tokens=False,
            )

            student_tokens = [
                self.tokenizer.decode(to_decode, skip_special_tokens=True)
                for to_decode in student_token_ids
            ]

            teacher_input = approved_tokens + student_tokens

            is_valid, error_id, teacher_intervention = self.verify_output(
                teacher_input,
                chat=chat,
                last_err_ix=error_id,
                student_str=decoded_output,
            )

            chat.add_message(
                part=teacher_intervention, source="assistant", model_role="teacher"
            )

            print(
                "Teacher's evaluation:",
                f"is valid: {is_valid}, error_id: {error_id}, teacher_intervention: {teacher_intervention}",
                "\n ------------- ",
                end="\n\n",
                flush=True,
            )

            student_eos = self.tokenizer.eos_token in student_tokens and is_valid
            print(f"Student EOS: {student_eos}", end="\n\n\n", flush=True)

        return decoded_output

    def set_teacher_system_prompt(self, chat: Chat):
        """
        Set the system prompt for the teacher.
        This includes clearing the teacher's chat of previous parts.

        :param: chat: Chat, the current chat for the sample

        :return:
        """
        # clear the teacher's chat
        if chat.messages["teacher"]:
            chat.messages["teacher"] = []

        teacher_sys_prompt = self.eval_prompt.format_teacher_sys(
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
