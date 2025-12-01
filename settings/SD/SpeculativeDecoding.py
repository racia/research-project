from __future__ import annotations

import copy
import re
import warnings
from typing import Any

import torch

from data.DataSaver import DataSaver
from inference.Prompt import Prompt
from inference.utils import Source, flatten
from interpretability.utils import InterpretabilityResult
from settings.Model import Model
from settings.Setting import Setting


class SpeculativeDecoding(Setting):
    """
    This class handles everything related to speculative decoding.
    """

    def __init__(
        self,
        student: Model,
        teacher: Model,
        total_tasks: int,
        total_parts: int,
        init_prompt: Prompt = None,
        eval_prompt: Prompt = None,
        resume_prompt: Prompt = None,
        samples_per_task: int = 5,
        multi_system: bool = True,
        saver: DataSaver = None,
        name: str = "sd",
    ):
        """
        Initialize the speculative decoding setting.

        The speculative decoding setting consists of a teacher model, a student model, and a tokenizer.
        The init_prompt is the initial prompt that is used to start the chain of thought of the student model.
        The resume_prompt is used to prompt the student model to resume the chain of thought with the corrections by
        the teacher. The eval_prompt is used to prompt the teacher to evaluate the chain of thought of the student
        model.

        :param teacher: The teacher model
        :param student: The student model
        :param init_prompt: the initial prompt for the student
        :param eval_prompt: the evaluation prompt for the teacher
        :param resume_prompt: the resume prompt for the student
        :param total_tasks: the number of tasks
        :param total_parts: the number of parts
        :param samples_per_task: the number of samples per task
        :param multi_system: whether the chat for one sample consists of multiple systems, i.e. a teacher and a student
        """
        super().__init__(
            model=student,
            total_tasks=total_tasks,
            total_parts=total_parts,
            init_prompt=init_prompt,
            samples_per_task=samples_per_task,
            multi_system=multi_system,
            saver=saver,
            name=name,
        )
        self.teacher: Model = teacher
        self.student: Model = student
        self.tokenizer = student.tokenizer

        self.init_prompt = init_prompt
        self.eval_prompt: Prompt = eval_prompt
        self.resume_prompt: Prompt = resume_prompt

        self.initial_student_output = None
        self.use_fallback = False
        self.student_eos = False

        # save the original chat so the refined output can be added later on
        self.chat = None
        self.curr_eval_dict = {
            "iterations": 0,
            "intervention_ix": [],
            "early_stop": False,
            "teacher_prob_approved_tokens": {},
            "intervention_with_prob": {},
            "teacher_empty_suggestion": 0,
            "max_supp_attn": [],
            "attn_on_target": [],
        }

    def generate_student(self) -> tuple[str, InterpretabilityResult]:
        """
        Generate some candidates using the student model.

        The student model generates a chain of thought based on the input string.
        The maximum length of chain of thought is handled by the max_length parameter.

        :return: The output of the student model
        """
        # message order: sys prompt + ex - part task - (stud out) - ! resume mes ! - (stud out iter)
        eval_prompt_match = re.match(
            self.resume_prompt.text,
            self.student.chat.messages[-1]["content"],
        )
        if eval_prompt_match:
            print("Remove the message with previous iteration!")
            self.student.chat.remove_message(-1)
        else:
            print("No resume message!")
            self.student.chat.add_message(
                part=self.resume_prompt.text,
                source=Source.user,
                ids=self.resume_prompt.ids,
            )

        self.student.chat.move_approved_message(
            self.teacher.chat,
            wrapper=self.resume_prompt.wrapper,
            source=Source.assistant,
        )
        print(
            "\n--------------\n",
            "Evaluation prompt of the user: ",
            self.resume_prompt.text,
            sep="\n",
            end="\n--------------\n\n",
        )

        print(
            "\n--------------\n",
            "Message that the student should continue: ",
            self.student.chat.messages[-1]["content"],
            sep="\n",
            end="\n--------------\n\n",
        )

        # Call with the whole chat
        decoded_output, interpretability = self.student.call(
            self.part, from_chat=True, to_continue=True, filter_eot=True
        )

        return decoded_output, interpretability

    def verify_output(
        self,
        student_message: dict,
        last_err_inx: int = -1,
    ) -> tuple[bool, int | None, str | None]:
        """
        Verify the candidates using the teacher model.

        Let the teacher go through the whole chain of thought proposed by the student.
        If the teacher disagrees at some point, return the teachers suggestion as the CoT step.

        :param student_message: the student message with string output, tokens, and ids
        :param last_err_inx: the index of the last error

        :return: A tuple containing a boolean indicating whether the current CoT is valid,
        an integer or None indicating the error index and the teacher's intervention or None
        """
        print(
            f"TEACHER CHAT BEFORE VERIFICATION",
            self.teacher.chat,
            end="\nEND OF TEACHER CHAT\n\n",
            flush=True,
        )

        approved_tokens = (
            list(self.curr_eval_dict["teacher_prob_approved_tokens"].keys()) or []
        )

        top_tokens_encoded = []
        suggested_token = None
        suggested_token_backup = None
        suggested_token_backup_encoded = None
        top_tokens_decoded_probs = None
        inx = 0
        print("student_message", student_message)
        if type(student_message["tokens"][0]) is not list:
            raise TypeError(
                f"Student message tokens are not a list of lists: {student_message['tokens']}"
            )

        flat_student_tokens = flatten(student_message["tokens"][-1])
        print(
            f"Student tokens {flat_student_tokens[:last_err_inx + 1]} have already been approved in a "
            f"previous iteration."
        )

        for inx, student_token in enumerate(flat_student_tokens[last_err_inx + 1 :]):
            print(
                f"TEACHER MESSAGES BEFORE VERIFICATION",
                self.teacher.chat.messages[-1],
                end="\nEND OF TEACHER MESSAGES\n\n",
                flush=True,
            )

            is_eos_token = (student_token == self.tokenizer.eos_token) or (
                student_token == "<|eot_id|>"
            )
            is_last_token = inx == len(flat_student_tokens)
            if is_eos_token or is_last_token:
                self.student_eos = True

            print(
                f"Verifying token '{student_token}' at index {inx + last_err_inx + 1}",
                end="\n\n",
                flush=True,
            )

            input_ids = self.teacher.chat.convert_into_datatype(
                datatype="ids",
                identify_target=False,
                to_continue=True,
            )

            teacher_probs = self.teacher.call_probs(input_ids.to("cuda"))

            if self.teacher.p:
                top_tokens_decoded_probs, top_tokens_encoded = self.get_top_p_tokens(
                    teacher_probs=teacher_probs, skip_eos=True
                )
            else:
                top_tokens_decoded_probs, top_tokens_encoded = self.get_top_k_tokens(
                    teacher_probs=teacher_probs, skip_eos=True
                )

            top_tokens_decoded = list(top_tokens_decoded_probs.keys())

            if len(top_tokens_decoded_probs) == 0:
                print(
                    f"Teacher model did not predict any token with cumulative prob >= {self.teacher.p} or k = "
                    f"{self.teacher.k}.",
                    f"\n Using the students token '{student_token}' as the teacher's suggestion.",
                    end="\n\n",
                    flush=True,
                )
                self.curr_eval_dict["teacher_empty_suggestion"] += 1
                suggested_token = student_token
                suggested_token_backup = student_token
            else:
                # teacher suggests token with highest prob = first token in dict
                suggested_token = str(top_tokens_decoded[0])
                if suggested_token is None:
                    raise ValueError(
                        f"Teacher suggested token is None. Teacher probs: {teacher_probs}"
                    )
                if len(top_tokens_decoded) > 1:
                    suggested_token_backup = str(top_tokens_decoded[1])
                    suggested_token_backup_encoded = top_tokens_encoded[1]
                else:
                    warnings.warn(
                        f"The teacher suggested only one token: {suggested_token}. Using this token as a "
                        f"backup as well."
                    )
                    suggested_token_backup = str(top_tokens_decoded[0])
                    suggested_token_backup_encoded = top_tokens_encoded[0]

            student_token_id = self.tokenizer.convert_tokens_to_ids(student_token)

            print(
                "Student's token and id:",
                student_token,
                student_token_id,
                sep=" ",
                end="\n\n",
            )

            # check decoded top tokens contain student token AND if student_token_id is in top token ids of teacher
            st_token_is_probable = student_token in top_tokens_decoded
            st_id_is_probable = student_token_id in top_tokens_encoded
            if not (st_token_is_probable or st_id_is_probable):
                # student token is not in top tokens
                break

            approved_tokens.append(student_token)
            self.teacher.chat.adjust_message(
                self.tokenizer.convert_tokens_to_string([student_token]),
                student_token_id,
            )
            print(
                "top_tokens_decoded_probs[student_token]",
                type(top_tokens_decoded_probs[student_token]),
                top_tokens_decoded_probs[student_token],
            )
            self.curr_eval_dict["teacher_prob_approved_tokens"][student_token] = (
                top_tokens_decoded_probs[student_token]
            )

            print(f"Teacher approved token {student_token}", end="\n\n", flush=True)

        # student's CoT is not empty and no disagreement was found
        on_last_token = inx == len(flat_student_tokens[last_err_inx + 1 :]) - 1
        if self.teacher_suggests_eos(suggested_token):
            if on_last_token:
                print(f"Teacher generated EOS", end="\n\n\n", flush=True)
                return True, None, None
            else:
                print(
                    f"Teacher generated EOS, but not on the last student token",
                    end="\n\n\n",
                    flush=True,
                )
                if suggested_token_backup is None or self.special_tokens_in(
                    [suggested_token_backup]
                ):
                    suggested_token_backup = student_token
                    suggested_token_backup_encoded = self.tokenizer.encode("")
                    print(
                        f"Using student token {student_token} as backup as the backup token is None or a special token",
                        end="\n\n\n",
                        flush=True,
                    )

                print(
                    f"Using the backup token '{suggested_token_backup}'",
                    end="\n\n\n",
                    flush=True,
                )
                self.teacher.chat.adjust_message(
                    self.tokenizer.convert_tokens_to_string([suggested_token_backup]),
                    suggested_token_backup_encoded,
                )

                # we need the overall index -> add the last error index
                ix = inx + last_err_inx + 1

                return False, ix, suggested_token_backup

        print(f"Teacher did not generate EOS", end="\n\n\n", flush=True)

        self.curr_eval_dict["intervention_ix"].append(int(inx))
        if top_tokens_decoded_probs:
            self.curr_eval_dict["intervention_with_prob"][suggested_token] = (
                top_tokens_decoded_probs[suggested_token]
            )

        self.teacher.chat.adjust_message(
            self.tokenizer.convert_tokens_to_string([suggested_token]),
            top_tokens_encoded[0],
        )

        return False, inx, suggested_token

    def get_top_k_tokens(
        self, teacher_probs, skip_eos=True
    ) -> tuple[dict[Any, Any], list[int]]:
        """
        Get the top k tokens from the teacher model that have a probability higher than min_prob.

        :param teacher_probs: the probabilities of the teacher model output
        :param skip_eos: boolean indicating whether to skip the EOS token

        :return: a tuple containing the top k tokens in decoded and encoded form
        """
        top_tokens_decoded_probs = {}
        top_tokens_encoded = []
        j = 0
        # some of the top tokens could be special tokens
        # we generate until we have k valid tokens
        while (
            len(top_tokens_decoded_probs) < self.teacher.k
            and (self.teacher.k + j) < teacher_probs.shape[-1]
            and j < 20
        ):
            top_probs, top_ids = torch.topk(teacher_probs, self.teacher.k + j, dim=-1)

            for token_id in top_ids[0]:
                decoded_token = self.tokenizer.convert_ids_to_tokens(token_id.item())

                if decoded_token not in top_tokens_decoded_probs.keys():
                    if skip_eos and len(decoded_token) == 0:
                        continue

                    top_tokens_decoded_probs[decoded_token] = top_probs[0][j]
                    top_tokens_encoded.append(int(token_id))

                if len(top_tokens_decoded_probs) == self.teacher.k:
                    break

            j += 1

        print(
            "Teacher's top k tokens:",
            top_tokens_decoded_probs.keys(),
            top_tokens_encoded,
            sep=" ",
            end="\n",
            flush=True,
        )

        return top_tokens_decoded_probs, top_tokens_encoded

    def get_top_p_tokens(
        self, teacher_probs, skip_eos=True
    ) -> tuple[dict[Any, Any], list[int]]:
        """
        Get the top p tokens of the teacher.

        :param teacher_probs: the probabilities of the teacher
        :param skip_eos: bool, whether to skip the end-of-sentence token

        :return: a tuple containing the top tokens in decoded and encoded form
        """
        top_tokens_decoded_prob = {}
        top_tokens_encoded = []

        sorted_probs, sorted_indices = torch.sort(teacher_probs[0], descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_token_mask = cumulative_probs <= self.teacher.p
        # always include the most likely token
        top_token_mask[0] = True

        for token_id, prob in zip(
            sorted_indices[top_token_mask], sorted_probs[top_token_mask]
        ):
            decoded_token = self.tokenizer.convert_ids_to_tokens(token_id.item())

            print(
                f"Token id: {token_id}, decoded token: '{decoded_token}', prob: {prob}"
            )

            if skip_eos and len(decoded_token) == 0:
                continue

            if decoded_token not in top_tokens_decoded_prob:
                top_tokens_decoded_prob[decoded_token] = prob
                top_tokens_encoded.append(token_id)

        if len(top_tokens_decoded_prob) == 0:
            print(
                f"Teacher model did not predict any token with cumulative prob >= {self.teacher.p}.",
                end="\n\n",
                flush=True,
            )

        print(
            f"Teacher's top tokens with cumulative prob >= {self.teacher.p}:",
            top_tokens_decoded_prob.keys(),
            top_tokens_encoded,
            sep=" ",
            end="\n",
            flush=True,
        )

        return top_tokens_decoded_prob, top_tokens_encoded

    def teacher_suggests_eos(self, suggested_token: str) -> bool:
        """
        Check if the teacher model would generate an end-of-sentence (EOS) token next.

        :param suggested_token: str, the token that the teacher suggests

        :return: boolean indicating whether the teacher model would generate an EOS token next
        """
        print(f"Checking if teacher would generate EOS next", end="\n\n\n", flush=True)
        # check if current suggested token is EOS
        if suggested_token in (
            "<|eot_id|>",
            "<|start_header_id|>",
            self.tokenizer.eos_token,
        ):
            return True

        # check if next token would be EOS
        # ASSUMPTION: the last student message is already in the chat and is unfinished
        input_ids = self.teacher.chat.convert_into_datatype(
            datatype="ids", identify_target=False
        )
        teacher_probs = self.teacher.call_probs(input_ids.to("cuda"))
        top_tokens_decoded, top_tokens_encoded = self.get_top_k_tokens(
            teacher_probs=teacher_probs, skip_eos=True
        )
        print(
            "Teacher's top tokens EOS check:",
            top_tokens_decoded,
            end="\n\n",
            flush=True,
        )

        detected_eos_token_id = self.tokenizer.eos_token_id in top_tokens_encoded
        if self.special_tokens_in(top_tokens_decoded) or detected_eos_token_id:
            return True
        return False

    def special_tokens_in(self, tokens) -> bool:
        """
        Checks if the EOS, EOT or PAD tokens are in the list of provided tokens.

        :param tokens: the list of tokens to inspect
        :return: whether the said tokens are in the provided list
        """
        if self.tokenizer.eos_token in tokens:
            return True
        if self.tokenizer.pad_token in tokens:
            return True
        if "<|eot_id|>" in tokens:
            return True
        if "<|start_header_id|>" in tokens:
            return True
        return False

    def check_repetition(self) -> bool:
        """
        Check if the last token in the student's output is repeated more than 3 times in the last 5 tokens.
        :return:
        """
        last_message = self.teacher.chat.messages[-1]
        flat_student_tokens = flatten(last_message["tokens"])
        if len(flat_student_tokens) > 5:
            last_token = flat_student_tokens[-1]
            count_last_token = flat_student_tokens[-5:].count(last_token)
            if count_last_token >= 3:
                print(
                    f"Last token {last_token} repeated {count_last_token} times, stopping speculative "
                    f"decoding and using fallback solution.",
                    end="\n\n\n",
                    flush=True,
                )
                self.use_fallback = True
                self.curr_eval_dict["early_stop"] = True
                return True
        return False

    def speculative_decode(
        self,
        student_out: str,
        is_valid: bool,
        error_ix: int | None,
    ) -> None:
        """
        Apply the speculative decoding on the output of the student.

        :param student_out: the student's output
        :param is_valid: boolean indicating whether the current CoT is valid
        :param error_ix: the error index

        :return: str, the speculative decoding output
        """
        # at this point, the teacher has evaluated once and potentially given a correction
        revs = 1

        # for the first SD iteration, add the length of the wrapper to the error index
        # subsequent iterations will have the wrapper in the chat already
        previous_resume_start = [
            "Here",
            "Ġis",
            "Ġthe",
            "Ġimproved",
            "Ġversion",
            "Ġof",
            "Ġthe",
            "Ġprevious",
            "Ġoutput",
            ":Ċ",
        ]
        if (
            len(self.student.chat.messages[-1]["tokens"]) == 1
            and previous_resume_start == self.student.chat.messages[-1]["tokens"][:10]
        ):
            print(
                f"Adjusting error index by the length of the wrapper: {error_ix}, wrapper length: "
                f": {flatten(self.resume_prompt.wrapper['wrapper']['before']['tokens'])}",
            )
            error_ix += len(
                flatten(self.resume_prompt.wrapper["wrapper"]["before"]["tokens"])
            )
            print(f"Adjusted error index: {error_ix}", end="\n\n", flush=True)

        # the student will have to continue from the suggested token -> we have not yet reached EOS
        self.student_eos = False

        while not is_valid and revs < 15 and not self.student_eos:
            print(
                f" --------------- SD iteration {revs} --------------- ",
                end="\n\n",
                flush=True,
            )
            self.curr_eval_dict["iterations"] += 1
            if type(student_out) is not str:
                raise TypeError("Student output is not a string:", student_out)

            # check for repetitions
            if self.check_repetition():
                break

            student_out, interpretability = self.generate_student()
            if type(student_out) is not str:
                raise ValueError(f"Student output is not a string: {student_out}")

            self.saver.save_sd_iteration(
                part=self.part,
                iteration=revs,
                student_message=student_out,
                interpretability=interpretability,
            )

            self.curr_eval_dict["max_supp_attn"].append(
                interpretability.max_supp_attn if interpretability else None
            )
            self.curr_eval_dict["attn_on_target"].append(
                interpretability.attn_on_target if interpretability else None
            )

            print(
                " ---- Student ---- \n",
                "Resumed output of student: \n",
                student_out,
                end="\n--------------\n\n",
                flush=True,
            )

            print(
                " ---- Teacher ---- ",
                end="\n\n",
                flush=True,
            )
            is_valid, error_ix, teacher_intervention = self.verify_output(
                student_message=self.student.chat.messages[-1],
                last_err_inx=error_ix,
            )

            print(
                "Teacher's evaluation:",
                f"is valid: {is_valid}, error_ix: {error_ix}, teacher_intervention: {teacher_intervention}",
                "\n ------------- ",
                end="\n\n",
                flush=True,
            )

            print(
                f"--------------------------------------------", end="\n\n", flush=True
            )
            revs += 1

        if not is_valid:
            self.student.chat.remove_message(-1)
            self.student.chat.move_approved_message(
                self.teacher.chat, wrapper=self.resume_prompt.wrapper
            )
        else:
            print(
                f"Teacher approved the students output after {revs - 1} iterations.",
                end="\n\n",
                flush=True,
            )

    def apply_setting(
        self, decoded_output: str
    ) -> tuple[str, dict, InterpretabilityResult]:
        """
        Run the speculative decoding for one instance.

        The speculative decoding consists of the following steps:
        1. Generate a chain of thought with the student model.
        2. Verify that chain of thought using the teacher model.
        3. If the teacher model disagrees, correct the chain of thought and continue with the student model.
        4. Repeat steps 2 and 3 until the teacher model agrees with the chain of thought.

        :param decoded_output: the current output of the student
        :return: The decoded output, the number of revisions and the interpretability result
        """
        self.initial_student_output = decoded_output
        original_student_chat = copy.deepcopy(self.student.chat)
        self.teacher.chat = self.create_teacher_chat(
            teacher_sys=self.eval_prompt,
            tokenizer=self.teacher.tokenizer,
            remove_last=True,
        )
        empty_message = {}
        teacher_message = self.eval_prompt.format_teacher_message(empty_message)
        self.teacher.chat.add_message(teacher_message)
        self.teacher.chat.part = self.part

        # reset evaluation dict for each part
        self.curr_eval_dict = {
            "iterations": 0,
            "intervention_ix": [],
            "early_stop": False,
            "teacher_prob_approved_tokens": {},
            "intervention_with_prob": {},
            "teacher_empty_suggestion": 0,
            "max_supp_attn": [],
            "attn_on_target": [],
        }

        print(
            " ------------- Starting speculative decoding ------------- ",
            end="\n\n\n",
            flush=True,
        )
        print(f" ---- SD iteration 0 ---- ", end="\n\n\n", flush=True)
        self.curr_eval_dict["iterations"] += 1

        is_valid, error_ix, teacher_intervention = self.verify_output(
            self.student.chat.messages[-1]
        )

        print(
            "Teacher's evaluation:",
            f"is valid: {is_valid}, error_ix: {error_ix}, teacher_intervention: {teacher_intervention}",
            "\n ------------- ",
            end="\n\n",
            flush=True,
        )

        if not is_valid:
            self.student.chat.remove_message(-1)
            self.student.chat.move_approved_message(
                self.teacher.chat, wrapper=self.resume_prompt.wrapper
            )

            self.speculative_decode(
                student_out=decoded_output,
                is_valid=is_valid,
                error_ix=error_ix,
            )

        teacher_wrapper = self.eval_prompt.wrapper["wrapper"]["before"]["content"]
        last_message = self.student.chat.messages[-1]
        if teacher_wrapper in last_message["content"]:
            raise ValueError(
                f"Teacher wrapper found in the last message of the student: {last_message['content']}"
            )

        if self.use_fallback:
            print(
                "Using fallback solution for the student output.",
                end="\n\n",
                flush=True,
            )
            decoded_output = self.initial_student_output
            # the initial output does not contain special tokens, so filtering is not necessary
        else:
            # remove special tokens from output
            decoded_output = last_message["content"]

        # change the last message of the student to the refined output
        original_student_chat.remove_message(-1)
        original_student_chat.move_approved_message(self.student.chat)
        self.student.chat = original_student_chat

        return (
            decoded_output,
            self.curr_eval_dict,
            self.get_after_interpretability(),
        )
