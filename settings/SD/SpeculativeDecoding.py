from __future__ import annotations

import copy
from typing import Any

import torch

from data.DataSaver import DataSaver
from inference.Chat import Chat
from inference.Prompt import Prompt
from inference.utils import Source, flatten
from interpretability.utils import InterpretabilityResult
from settings.Model import Model
from settings.SD.utils import check_match
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
    ):
        """
        Initialize the speculative decoding setting.

        The speculative decoding setting consists of a teacher model, a student model, and a tokenizer.
        The init_prompt is the initial prompt that is used to start the chain of thought of the student model.
        The resume_prompt is used to prompt the student model to resume the chain of thought with the corrections by
        the teacher. The eval_prompt is used to prompt the teacher to evaluate the chain of thought of the student model.

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

        # the model sometimes predicts only parts of the word, e.g. "d" + "aniel" instead of daniel
        self.affix = False

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

        :return: the output of the student model
        """
        # remove the last message the student generated as it is only partially correct
        self.student.chat.remove_message(-1)
        # add the resume prompt to the student chat
        self.student.chat.add_message(
            part=self.resume_prompt.text, source=Source.user, ids=self.resume_prompt.ids
        )
        # teacher intervened and student should continue approved message == last message of teacher chat
        self.student.chat.move_approved_message(
            self.teacher.chat, wrapper=self.resume_prompt.wrapper
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
            self.part, from_chat=True, subfolder="iterations", to_continue=True
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
        approved_tokens = (
            list(self.curr_eval_dict["teacher_prob_approved_tokens"].keys()) or []
        )
        print(
            f"Student tokens {student_message['tokens'][:last_err_inx + 1]} have already been approved in a previous "
            f"iteration."
        )

        top_tokens_encoded = []
        suggested_token = None
        suggested_token_affix = None
        top_tokens_decoded_probs = None
        inx = 0
        print('student_message["tokens"]', student_message["tokens"])
        flat_student_tokens = flatten(student_message["tokens"])
        print("flat_student_tokens", flat_student_tokens)

        for inx, student_token in enumerate(flat_student_tokens[last_err_inx + 1 :]):
            # handle empty tokens by setting it to whitespace
            # TODO: remove?
            print(
                "Encoded token",
                student_token,
                self.tokenizer.convert_tokens_to_ids(student_token),
            )
            # student_token = student_token.lower().strip() or " "
            is_eos_token = student_token == self.tokenizer.eos_token
            is_last_token = inx == len(flat_student_tokens)
            if is_eos_token or is_last_token:
                self.student_eos = True

            # TODO: check if needed?
            # if self.affix:
            #     student_token = flat_student_tokens[inx - 1] + student_token
            #     student_token = student_token.strip()
            #     print(
            #         f"Verifying token combined with previous token {student_token} at indices {inx - 1} and {inx}",
            #         end="\n\n",
            #         flush=True,
            #     )
            # else:
            print(
                f"Verifying token '{student_token}' at index {inx}",
                end="\n\n",
                flush=True,
            )

            # if inx > 0 or last_err_inx > 0:
            #     print(
            #         f"Tokens accepted by the teacher in this iteration so far: {student_message['tokens'][:inx]}",
            #         end="\n\n",
            #         flush=True,
            #     )
            #
            #     _, student_out_approved = check_match(
            #         tokens=all_student_tokens[: last_err_inx + 1],
            #         string=all_student_str,
            #         inx=inx,
            #         intervention=all_student_tokens[last_err_inx + 1],
            #     )
            # # first iteration -> teacher gets no student output
            # else:
            #     student_out_approved = " "

            input_ids = self.teacher.chat.convert_into_datatype(
                datatype="ids", identify_target=False, to_continue=True
            )
            print(
                f"TEACHER Formatted prompt (to remove):",
                self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0],
                sep="\n",
                end="\n",
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
            else:
                # teacher suggests token with highest prob = first token in dict
                suggested_token = str(top_tokens_decoded[0])
            print("suggested_token", suggested_token)

            student_token_id = (
                self.tokenizer.convert_tokens_to_ids(student_token) or None
            )

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

                # if the student token was considered as an affix and is not in the top tokens,
                # return the suggested token of the previous step
                if self.affix and suggested_token_affix:
                    self.affix = False
                    return False, inx - 1, suggested_token_affix

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

            self.affix = False
            print(f"Teacher approved token {student_token}", end="\n\n", flush=True)

        # student's CoT is not empty and no disagreement was found
        on_last_token = inx == len(flatten(student_message["tokens"])) - 1
        if self.teacher_suggests_eos(suggested_token) and on_last_token:
            print(f"Teacher generated EOS", end="\n\n\n", flush=True)
            return True, None, None

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
                decoded_token = self.tokenizer.convert_ids_to_tokens(
                    token_id.item()
                )  # .lower().strip()

                if decoded_token not in top_tokens_decoded_probs.keys():
                    if skip_eos and len(decoded_token) < 1:
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
            decoded_token = self.tokenizer.convert_ids_to_tokens(
                token_id.items()
            )  # .lower().strip()

            print(f"Token id: {token_id}, decoded token: {decoded_token}, prob: {prob}")

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
        if suggested_token in ("<|eot_id|>", self.tokenizer.eos_token):
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

        # formatted_eval_prompt = self.prepare_prompt(chat=teacher_chat, resume_gen=True)
        # input_ids = self.tokenizer.encode(
        #     formatted_eval_prompt, return_tensors="pt", add_special_tokens=True
        # ).to("cuda:1")
        #
        # probs = self.teacher.call_probs(input_ids=input_ids)
        # top_tokens_decoded, top_tokens_encoded = self.get_top_k_tokens(
        #     teacher_probs=probs, k=5, skip_eos=False
        # )
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
        return False

    def prepare_prompt(self, chat: Chat, resume_gen: bool = False) -> str:
        """
        Prepares the prompt to include the current part of the sample.

        :param chat: the current chat
        :param resume_gen: whether to resume generation from the last message

        :return: prompt with the task and the current part
        """
        if self.model.to_continue or resume_gen:
            formatted_prompt = self.model.tokenizer.apply_chat_template(
                chat.messages, tokenize=False, continue_final_message=True
            )
        else:
            formatted_prompt = self.model.tokenizer.apply_chat_template(
                chat.messages, tokenize=False, add_generation_prompt=True
            )

        return formatted_prompt

    def get_previous_approved_output(
        self,
        student_out: str,
        teacher_intervention: str,
        error_id: int,
        prev_output: list,
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
        # TODO: check if tokenization works
        # student_tokens = self.string_to_tokens(model_out=student_out)
        student_tokens = self.tokenizer.tokenize(student_out)
        if type(student_out) is not str:
            raise TypeError(f"Student output is not a string: {student_out}")

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
            corrected_tokens, corrected_str = check_match(
                tokens=student_chain,
                string=student_complete_out,
                inx=error_id,
                intervention=teacher_intervention,
            )
        else:
            corrected_tokens = [teacher_intervention]
            corrected_str = teacher_intervention

        return corrected_tokens, corrected_str

    def speculative_decode(
        self,
        student_out: str,
        is_valid: bool,
        error_id: int | None,
    ) -> None:
        """
        Apply the speculative decoding on the output of the student.

        :param student_out: the student's output
        :param is_valid: boolean indicating whether the current CoT is valid
        :param error_id: the error index

        :return: str, the speculative decoding output
        """
        print(
            " ---- Teacher ---- ",
            end="\n\n",
            flush=True,
        )

        # at this point, the teacher has evaluated once and potentially given a correction
        revs = 1
        # corrected_tokens = []
        # corrected_str = ""
        # interpretability = None

        while (
            not is_valid and revs < 15 and not self.student_eos
        ):  # TODO: decide on max revisions
            print(
                f" --------------- SD iteration {revs} --------------- ",
                end="\n\n",
                flush=True,
            )
            self.curr_eval_dict["iterations"] += 1
            if type(student_out) is not str:
                raise TypeError("Student output is not a string:", student_out)

            # corrected_tokens, corrected_str = self.get_previous_approved_output(
            #     student_out=student_out,
            #     teacher_intervention=teacher_intervention,
            #     error_id=error_id,
            #     prev_str=corrected_str,
            #     prev_output=corrected_tokens,
            # )

            # check for repetitions
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
                    break

            print("TEACHER CHAT", self.teacher.chat, end="\n\n", flush=True)

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
            # self.student.chat.add_message(part=student_out, source="assistant")

            is_valid, error_id, teacher_intervention = self.verify_output(
                student_message=self.student.chat.messages[-1],
                # approved_tokens=corrected_tokens,
                # approved_str=corrected_str,
                last_err_inx=error_id,
            )

            # # only add teacher intervention to chat if the teacher disagrees
            # # otherwise the teacher doesn't propose anything
            # if not is_valid:
            #     if teacher_intervention:
            #         self.student.chat.add_message(
            #             part=teacher_intervention,
            #             source=Source.assistant,
            #         )
            #     else:
            #         self.student.chat.add_message(part=" ", source=Source.assistant)

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

        if not is_valid:
            self.student.chat.remove_message(-1)
            self.student.chat.move_approved_message(self.teacher.chat)

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
        # TODO: teacher tokens in the student chat?
        # TODO: the iterations is cut off at the middle of the message and teacher message is saved instead of the teacher one:
        # The student's response was:
        #
        #
        #
        # Reasoning: Daniel
        self.initial_student_output = decoded_output
        original_student_chat = copy.deepcopy(self.student.chat)
        self.teacher.chat = self.create_teacher_chat(
            teacher_sys=self.eval_prompt,
            tokenizer=self.teacher.tokenizer,
            remove_last=True,
        )
        empty_message = {
            "content": "\n\n",
            "original_content": "\n\n",
            "tokens": ["Ċ"],
            "ids": [271],
            "spans_with_types": {(0, 1): "ans"},
        }
        teacher_message = self.eval_prompt.format_teacher_message(empty_message)
        self.teacher.chat.add_message(teacher_message)

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

        is_valid, error_id, teacher_intervention = self.verify_output(
            self.student.chat.messages[-1]
        )

        print(
            "Teacher's evaluation:",
            f"is valid: {is_valid}, error_id: {error_id}, teacher_intervention: {teacher_intervention}",
            "\n ------------- ",
            end="\n\n",
            flush=True,
        )

        interpretability = None
        if not is_valid:
            # # if teacher suggests a token, add it to its chat
            # self.teacher.chat.add_message(
            #     part=teacher_intervention, source=Source.assistant
            # )

            self.speculative_decode(
                student_out=decoded_output,
                is_valid=is_valid,
                error_id=error_id,
            )

        if self.use_fallback:
            decoded_output = self.initial_student_output

        teacher_wrapper = "The student's response was:"
        last_message = self.student.chat.messages[-1]
        if teacher_wrapper in last_message["content"]:
            raise ValueError(
                f"Teacher wrapper found in the last message of the student: {last_message['content']}"
            )

        # change the last message of the student to the refined output
        original_student_chat.remove_message(-1)
        original_student_chat.move_approved_message(self.student.chat)
        self.student.chat = original_student_chat

        return decoded_output, self.curr_eval_dict, self.get_after_interpretability()

    # def string_to_tokens(self, model_out: str) -> list[str]:
    #     """
    #     Turn a models output into a list of tokens by encoding and decoding it.
    #
    #     :param model_out: the student's output
    #
    #     :return: list of tokens
    #     """
    #     student_token_ids = self.tokenizer.encode(
    #         model_out,
    #         add_special_tokens=False,
    #     )
    #
    #     student_tokens = [
    #         self.tokenizer.decode(to_decode, skip_special_tokens=True)
    #         for to_decode in student_token_ids
    #     ]
    #     torch.cuda.empty_cache()
    #
    #     return student_tokens
