from __future__ import annotations

import copy
import re

import torch

from inference.Chat import Chat, Source
from inference.Prompt import Prompt
from interpretability.Interpretability import Interpretability
from settings.Model import Model
from settings.Setting import Setting
from settings.config import Wrapper
from settings.utils import Enumerate


def check_match(tokens, string, ix=None, intervention=None) -> tuple[list, str]:
    """
    Check if the token list matches the string.

    :param tokens: list of tokens that should be checked for a match
    :param string: the string the tokens are matched against
    :param ix: the index up until which the tokens are approved/index of first error
    :param intervention: the intervention that should be added to the string

    :return: Tuple(list, str): the tokens and the string
    """
    if not ix or ix >= len(tokens):
        ix = len(tokens)

    out_tokens = [token.strip() if token else token for token in tokens][:ix]
    pattern = r"\s*" + r"\s*".join(map(re.escape, out_tokens))

    match = re.match(pattern, string)

    if match:
        out_string = str(match.group(0))
    else:
        out_string = " ".join(out_tokens)

    if intervention:
        out_string += (
            " " + intervention
            if intervention.isalpha() and not (intervention in ["ing", "ed", "s"])
            else intervention
        )
        out_tokens = out_tokens + [intervention]

    print(
        f"Checking match for tokens: {tokens[:ix]} and string: {string}. Result: {match}",
        end="\n\n",
        flush=True,
    )

    return out_tokens, out_string


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
        interpretability: Interpretability,
        init_prompt: Prompt = None,
        eval_prompt: Prompt = None,
        resume_prompt: Prompt = None,
        samples_per_task: int = 5,
        multi_system: bool = True,
        wrapper: Wrapper = None,
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
        :param interpretability: optional interpretability instance
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
        self.teacher: Model = teacher
        self.student: Model = student
        self.tokenizer = student.tokenizer

        self.init_prompt = init_prompt
        self.eval_prompt: Prompt = eval_prompt
        self.resume_prompt: Prompt = resume_prompt

        self.initial_student_output = None
        self.student_eos = False
        self.teacher_eos = False

        # the model sometimes predicts only parts of the word, e.g. "d" + "aniel" instead of daniel
        self.affix = False

        # save the original chat so the refined output can be added later on
        self.chat = None

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
        message_to_continue = self.resume_prompt.format_resume_message(corrected_in)
        chat.add_message(message_to_continue, source="assistant", model_role="student")
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
        self,
        student_out_str: str,
        chat: Chat,
        k=10,
        last_err_ix=-1,
        approved_tokens=None,
        approved_str=None,
    ) -> tuple[bool, int | None, str | None]:
        """
        Verify the candidates using the teacher model.

        Let the teacher go through the whole chain of thought proposed by the student.
        If the teacher disagrees at some point, return the teachers suggestion as the CoT step.

        :param student_out_str: the output of the student
        :param chat: the current chat
        :param k: the number of top candidates to consider from the teacher model
        :param last_err_ix: the index of the last error
        :param approved_tokens: the tokens that have been approved by the teacher so far
        :param approved_str: the string that has been approved by the teacher so far

        :return: A tuple containing a boolean indicating whether the current CoT is valid,
        an integer or None indicating the error index and the teacher's intervention or None
        """
        student_tokens = self.string_to_tokens(model_out=student_out_str)
        if approved_tokens:
            all_student_tokens = approved_tokens + student_tokens
            print(
                f"Student tokens {all_student_tokens[:last_err_ix + 1]} have already been approved in a previous "
                f"iteration."
            )
            all_student_str = approved_str + student_out_str
        else:
            all_student_tokens = student_tokens
            all_student_str = student_out_str
            approved_tokens = []

        suggested_token = None
        ix = 0

        for ix, student_token in enumerate(all_student_tokens[last_err_ix + 1 :]):
            if student_token:
                student_token = student_token.lower().strip()
                if student_token == self.tokenizer.eos_token or ix == len(
                    all_student_tokens
                ):
                    self.student_eos = True
            else:
                # handle empty tokens by setting it to whitespace
                student_token = " "

            if self.affix:
                student_token = all_student_tokens[ix - 1] + student_token
                student_token = student_token.strip()
                print(
                    f"Verifying token combined with previous token {student_token}",
                    end="\n\n",
                    flush=True,
                )
            else:
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

                _, student_out_approved = check_match(
                    tokens=approved_tokens[:last_err_ix],
                    string=all_student_str,
                    ix=ix,
                    intervention=approved_tokens[last_err_ix],
                )
            # first iteration -> teacher gets no student output
            else:
                student_out_approved = " "

            # the prompt is added into the chat in this method
            teacher_message = self.eval_prompt.format_teacher_message(
                student_out_approved
            )
            chat.add_message(part=teacher_message, model_role="teacher", source="user")
            print(
                "\n--------------\n",
                "Formatted teacher prompt:",
                teacher_message,
                sep="\n",
                end="\n--------------\n\n",
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
            print("suggested_token", suggested_token)

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
                # if the student token was considered as an affix and is not in the top tokens,
                # return the suggested token of the previous step
                if self.affix is True:
                    self.affix = False
                    return False, ix - 1, suggested_token_affix

                # check if token could be an affix
                if len(student_token) < 4:
                    self.affix = True
                    # save the current suggested token in case the token is not an affix
                    suggested_token_affix = suggested_token
                    print(
                        f"Potential affix found: {student_token}",
                        end="\n\n",
                        flush=True,
                    )
                    continue

                # student token is not in top tokens and is not an affix
                break

            approved_tokens.append(student_token)
            self.affix = False
            print(f"Teacher approved token {student_token}", end="\n\n", flush=True)

        if suggested_token == ("<|eot_id|>" or self.tokenizer.eos_token):
            self.teacher_eos = True

        self.teacher_eos = self.check_if_teacher_eos(chat, suggested_token)

        # student's CoT is not empty and no disagreement was found
        if self.teacher_eos and ix == len(student_tokens) - 1:
            print(f"Teacher generated EOS", end="\n\n\n", flush=True)
            return True, None, None
        print(f"Teacher did not generate EOS", end="\n\n\n", flush=True)
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
                    if skip_eos and len(decoded_token) < 1:
                        continue
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
        print(
            "Teacher's top tokens EOS check:",
            top_tokens_decoded,
            end="\n\n",
            flush=True,
        )

        if (
            self.tokenizer.eos_token or self.tokenizer.pad_token or "<|eot_id|>"
        ) in top_tokens_decoded or self.tokenizer.eos_token_id in top_tokens_encoded:
            return True
        return False

    def prepare_prompt(
        self,
        chat: Chat,
        resume_gen=False,
        model_role="student",
    ) -> str:
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

    def get_previous_approved_output(
        self,
        prev_output: list,
        student_out: str,
        error_id: int,
        teacher_intervention: str,
        prev_str: str = None,
    ) -> tuple[list, str]:
        """
        Get the output that the teacher approved up until now.
        This includes suggested tokens by the teacher.

        :param prev_output: the current student chain
        :param student_out: the student's output
        :param error_id: the error id as returned by the teacher
        :param teacher_intervention: the suggested token by the teacher
        :param prev_str: the previous approved string

        :return: the corrected chain as a list and the corrected chain as a string
        """
        student_tokens = self.string_to_tokens(model_out=student_out)

        if prev_output:
            prev_output_tokens = [token.strip() for token in prev_output]
            student_chain = prev_output_tokens + student_tokens
        else:
            student_chain = student_tokens

        if prev_str:
            student_complete_out = prev_str + student_out
        else:
            student_complete_out = student_out

        if len(student_chain) >= 1:
            corrected_chain, corrected_str = check_match(
                tokens=student_chain,
                string=student_complete_out,
                ix=error_id,
                intervention=teacher_intervention,
            )
        else:
            corrected_chain = [teacher_intervention]
            corrected_str = teacher_intervention

        return corrected_chain, corrected_str

    def apply_setting(self, decoded_output: str, chat: Chat = None) -> str:
        """
        Run the speculative decoding for one instance.

        The speculative decoding consists of the following steps:
        1. Generate a chain of thought with the student model.
        2. Verify that chain of thought using the teacher model.
        3. If the teacher model disagrees, correct the chain of thought and continue with the student model.
        4. Repeat steps 2 and 3 until the teacher model agrees with the chain of thought.

        :param decoded_output: the current output of the student
        :param chat: the current chat, only necessary in the SD and feedback setting

        :return: The decoded output
        """
        # save the initial student output as a fallback solution
        self.initial_student_output = decoded_output
        self.set_teacher_system_prompt(chat=chat)
        chat = self.create_chat_copy(chat=chat)

        print(
            " ------------- Starting speculative decoding ------------- ",
            end="\n\n\n",
            flush=True,
        )
        print(f" ---- SD iteration 1 ---- ", end="\n\n\n", flush=True)

        is_valid, error_id, teacher_intervention = self.verify_output(
            student_out_str=decoded_output, chat=chat
        )

        print(
            "Teacher's evaluation:",
            f"is valid: {is_valid}, error_id: {error_id}, teacher_intervention: {teacher_intervention}",
            "\n ------------- ",
            end="\n\n",
            flush=True,
        )

        if not is_valid:
            # if teacher suggests a token, add it to its chat
            chat.add_message(
                part=teacher_intervention, source="assistant", model_role="teacher"
            )

            decoded_output = self.speculative_decode(
                student_out=decoded_output,
                is_valid=is_valid,
                error_id=error_id,
                teacher_intervention=teacher_intervention,
                chat=chat,
            )

        # change the last message of the student to the refined output
        self.chat.messages["student"][-1]["content"] = decoded_output

        return decoded_output

    def speculative_decode(
        self,
        student_out: str,
        is_valid: bool,
        error_id: int | None,
        teacher_intervention: str | None,
        chat: Chat,
    ) -> dict[str, str | None] | str:
        """
        Apply the speculative decoding on the output of the student.

        :param student_out: the student's output
        :param is_valid: boolean indicating whether the current CoT is valid
        :param error_id: the error index
        :param teacher_intervention: the teacher's suggestion
        :param chat: the current chat

        :return: str, the speculative decoding output
        """
        print(
            " ---- Teacher ---- ",
            end="\n\n",
            flush=True,
        )

        # at this point, the teacher has evaluated once and potentially given a correction
        revs = 2
        corrected_chain = []
        corrected_str = ""

        while (
            not is_valid and revs < 15 and not self.student_eos
        ):  # TODO: decide on max revisions
            print(
                f" --------------- SD iteration {revs} --------------- ",
                end="\n\n",
                flush=True,
            )
            corrected_chain, corrected_str = self.get_previous_approved_output(
                prev_output=corrected_chain,
                error_id=error_id,
                teacher_intervention=teacher_intervention,
                student_out=student_out,
                prev_str=corrected_str,
            )

            # check for repetitions
            if len(corrected_chain) > 5:
                count_last_token = corrected_chain[-5:].count(corrected_chain[-1])
                if count_last_token >= 3:
                    print(
                        f"Last token {corrected_chain[-1]} repeated {count_last_token} times, stopping speculative "
                        f"decoding",
                        end="\n\n\n",
                        flush=True,
                    )
                    break

            student_out = self.generate_student(corrected_in=corrected_str, chat=chat)
            print(
                " ---- Student ---- \n",
                "Resumed output of student: \n",
                student_out,
                end="\n--------------\n\n",
                flush=True,
            )
            chat.add_message(part=student_out, source="assistant", model_role="student")

            is_valid, error_id, teacher_intervention = self.verify_output(
                student_out_str=student_out,
                approved_tokens=corrected_chain,
                approved_str=corrected_str,
                chat=chat,
                last_err_ix=error_id,
            )

            # only add teacher intervention to chat if the teacher disagrees
            # otherwise the teacher doesn't propose anything
            if not is_valid:
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

            print(
                f"--------------------------------------------", end="\n\n", flush=True
            )

            revs += 1

        return corrected_str + student_out

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

    def string_to_tokens(self, model_out: str) -> list[str]:
        """
        Turn a models output into a list of tokens by encoding and decoding it.

        :param model_out: the student's output

        :return: list of tokens
        """
        student_token_ids = self.tokenizer.encode(
            model_out,
            add_special_tokens=False,
        )

        student_tokens = [
            self.tokenizer.decode(to_decode, skip_special_tokens=True)
            for to_decode in student_token_ids
        ]

        return student_tokens

    def create_chat_copy(self, chat: Chat) -> Chat:
        """
        Create a copy of the chat.

        :param chat: Chat, the chat that should be copied

        :return: Chat, the copied chat
        """
        self.chat = chat
        sd_chat = copy.deepcopy(chat)

        return sd_chat
