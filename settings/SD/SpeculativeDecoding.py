from __future__ import annotations

import copy
import re

import torch

from data.DataSaver import DataSaver
from inference.Chat import Chat
from inference.Prompt import Prompt
from inference.utils import Source
from interpretability.utils import InterpretabilityResult
from settings.Model import Model
from settings.Setting import Setting


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

        self.student_eos = False

        # the model sometimes predicts only parts of the word, e.g. "d" + "aniel" instead of daniel
        self.affix = False

    def generate_student(
        self, corrected_str: str, corrected_tokens: list[str]
    ) -> tuple[str, InterpretabilityResult]:
        """
        Generate some candidates using the student model.

        The student model generates a chain of thought based on the input string.
        The maximum length of this chain of thought is handled by the max_length parameter.

        :param corrected_str: the correct part of the student's previous output with the teacher's suggestion
        :param corrected_tokens: the corresponding ids
        :return: The output of the student model
        """
        self.student.chat.add_message(part=self.resume_prompt.text, source=Source.user)
        print(
            "\n--------------\n",
            "Evaluation prompt of the user: ",
            self.resume_prompt.text,
            sep="\n",
            end="\n--------------\n\n",
        )

        # add the output the student should continue as an assistant message
        # TODO: Should not include eos_token_id or other end tokens at the end!
        message_to_continue = self.resume_prompt.format_resume_message(
            corrected_str, corrected_tokens
        )
        self.student.chat.add_message(**message_to_continue)

        print(
            "\n--------------\n",
            "Message that the student should continue: ",
            message_to_continue,
            sep="\n",
            end="\n--------------\n\n",
        )

        # formatted_prompt = self.prepare_prompt(chat=self.student.chat, resume_gen=True)

        # Call with the whole chat
        return self.student.call(self.part, from_chat=True)

    def verify_output(
        self,
        student_message: dict,
        k: int = 10,
        last_err_inx: int = -1,
        approved_tokens: list = None,
        approved_str: str = None,
    ) -> tuple[bool, int | None, str | None]:
        """
        Verify the candidates using the teacher model.

        Let the teacher go through the whole chain of thought proposed by the student.
        If the teacher disagrees at some point, return the teachers suggestion as the CoT step.

        :param student_message: the student message with string output, tokens, and ids
        :param k: the number of top candidates to consider from the teacher model
        :param last_err_inx: the index of the last error
        :param approved_tokens: the tokens that have been approved by the teacher so far
        :param approved_str: the string that has been approved by the teacher so far

        :return: A tuple containing a boolean indicating whether the current CoT is valid,
        an integer or None indicating the error index and the teacher's intervention or None
        """
        suggested_token_affix = ""
        approved_tokens = approved_tokens or []
        all_student_tokens = approved_tokens + student_message["tokens"]
        all_student_str = approved_str or "" + student_message["content"]
        print(
            f"Student tokens {all_student_tokens[:last_err_inx + 1]} have already been approved in a previous "
            f"iteration."
        )

        suggested_token = None
        inx = 0

        for inx, student_token in enumerate(all_student_tokens[last_err_inx + 1 :]):
            # handle empty tokens by setting it to whitespace
            student_token = student_token.lower().strip() or " "
            is_eos_token = student_token == self.tokenizer.eos_token
            is_last_token = inx == len(all_student_tokens)
            if is_eos_token or is_last_token:
                self.student_eos = True

            if self.affix:
                student_token = all_student_tokens[inx - 1] + student_token
                student_token = student_token.strip()
                print(
                    f"Verifying token combined with previous token {student_token} at indices {inx - 1} and {inx}",
                    end="\n\n",
                    flush=True,
                )
            else:
                print(
                    f"Verifying token '{student_token}' at index {inx}",
                    end="\n\n",
                    flush=True,
                )

            if inx > 0 or last_err_inx > 0:
                print(
                    f"Tokens accepted by the teacher in this iteration so far: {student_message['tokens'][:inx]}",
                    end="\n\n",
                    flush=True,
                )

                _, student_out_approved = check_match(
                    tokens=all_student_tokens[: last_err_inx + 1],
                    string=all_student_str,
                    ix=inx,
                    intervention=all_student_tokens[last_err_inx + 1],
                )
            # first iteration -> teacher gets no student output
            else:
                student_out_approved = " "

            # the prompt is added into the chat in this method
            teacher_message = self.eval_prompt.format_teacher_message(student_message)
            self.teacher.chat.add_message(**teacher_message)

            # formatted_eval_prompt = self.prepare_prompt(
            #     chat=self.teacher.chat, resume_gen=True
            # )
            # input_ids = self.tokenizer.encode(
            #     formatted_eval_prompt, return_tensors="pt", add_special_tokens=True
            # ).to("cuda:1")

            input_ids = self.teacher.chat.convert_into_ids(identify_target=False).to(
                "cuda:1"
            )
            teacher_probs = self.teacher.call_probs(input_ids)
            top_tokens_decoded, top_tokens_encoded = self.get_top_k_tokens(
                teacher_probs=teacher_probs, k=k, skip_eos=True
            )

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
                    suggested_token_affix = (
                        all_student_tokens[last_err_inx - 1] + suggested_token
                    )
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
            self.affix = False
            print(f"Teacher approved token {student_token}", end="\n\n", flush=True)

        # student's CoT is not empty and no disagreement was found
        on_last_token = inx == len(student_message["tokens"]) - 1
        if self.teacher_suggests_eos(suggested_token) and on_last_token:
            print(f"Teacher generated EOS", end="\n\n\n", flush=True)
            return True, None, None
        print(f"Teacher did not generate EOS", end="\n\n\n", flush=True)
        return False, inx, suggested_token

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
        input_ids = self.teacher.chat.convert_into_ids(identify_target=False).to(
            "cuda:1"
        )
        teacher_probs = self.teacher.call_probs(input_ids)
        top_tokens_decoded, top_tokens_encoded = self.get_top_k_tokens(
            teacher_probs=teacher_probs, k=5, skip_eos=True
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
                ix=error_id,
                intervention=teacher_intervention,
            )
        else:
            corrected_tokens = [teacher_intervention]
            corrected_str = teacher_intervention

        return corrected_tokens, corrected_str

    def apply_setting(
        self, decoded_output: str
    ) -> tuple[str, int, InterpretabilityResult]:
        """
        Run the speculative decoding for one instance.

        The speculative decoding consists of the following steps:
        1. Generate a chain of thought with the student model.
        2. Verify that chain of thought using the teacher model.
        3. If the teacher model disagrees, correct the chain of thought and continue with the student model.
        4. Repeat steps 2 and 3 until the teacher model agrees with the chain of thought.

        :param decoded_output: the current output of the student

        :return: The decoded output
        """
        # TODO: save model output as strings, not tokens
        # TODO: save iterations
        # TODO: teacher tokens in the student chat?

        original_student_chat = copy.deepcopy(self.student.chat)
        self.teacher.chat = self.create_teacher_chat(
            teacher_sys=self.eval_prompt, tokenizer=self.student.tokenizer
        )

        print(
            " ------------- Starting speculative decoding ------------- ",
            end="\n\n\n",
            flush=True,
        )
        print(f" ---- SD iteration 1 ---- ", end="\n\n\n", flush=True)

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
            # if teacher suggests a token, add it to its chat
            self.teacher.chat.add_message(
                part=teacher_intervention, source=Source.assistant
            )

            decoded_output, interpretability = self.speculative_decode(
                student_out=decoded_output,
                is_valid=is_valid,
                error_id=error_id,
                teacher_intervention=teacher_intervention,
            )

        # change the last message of the student to the refined output
        self.student.chat = original_student_chat
        self.student.chat.messages[-1]["content"] = decoded_output

        return decoded_output, 0, interpretability

    def speculative_decode(
        self,
        student_out: str,
        is_valid: bool,
        error_id: int | None,
        teacher_intervention: str | None,
    ) -> tuple[dict[str, str | None] | str, InterpretabilityResult]:
        """
        Apply the speculative decoding on the output of the student.

        :param student_out: the student's output
        :param is_valid: boolean indicating whether the current CoT is valid
        :param error_id: the error index
        :param teacher_intervention: the teacher's suggestion

        :return: str, the speculative decoding output
        """
        print(
            " ---- Teacher ---- ",
            end="\n\n",
            flush=True,
        )

        # at this point, the teacher has evaluated once and potentially given a correction
        revs = 1
        corrected_tokens = []
        corrected_str = ""
        interpretability = None

        while (
            not is_valid and revs < 15 and not self.student_eos
        ):  # TODO: decide on max revisions
            print(
                f" --------------- SD iteration {revs} --------------- ",
                end="\n\n",
                flush=True,
            )
            corrected_tokens, corrected_str = self.get_previous_approved_output(
                student_out=student_out,
                teacher_intervention=teacher_intervention,
                error_id=error_id,
                prev_str=corrected_str,
                prev_output=corrected_tokens,
            )

            # check for repetitions
            if len(corrected_tokens) > 5:
                count_last_token = corrected_tokens[-5:].count(corrected_tokens[-1])
                if count_last_token >= 3:
                    print(
                        f"Last token {corrected_tokens[-1]} repeated {count_last_token} times, stopping speculative "
                        f"decoding",
                        end="\n\n\n",
                        flush=True,
                    )
                    break

            # TODO: save iterations of interpretability
            student_out, interpretability = self.generate_student(
                corrected_str, corrected_tokens
            )
            if type(student_out) is not str:
                raise ValueError(f"Student output is not a string: {student_out}")
            self.saver.save_sd_iteration(
                part=self.part,
                iteration=revs,
                student_message=student_out,
                interpretability=interpretability,
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
                approved_tokens=corrected_tokens,
                approved_str=corrected_str,
                last_err_inx=error_id,
            )

            # only add teacher intervention to chat if the teacher disagrees
            # otherwise the teacher doesn't propose anything
            if not is_valid:
                if teacher_intervention:
                    self.student.chat.add_message(
                        part=teacher_intervention,
                        source=Source.assistant,
                    )
                else:
                    self.student.chat.add_message(part=" ", source=Source.assistant)

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

        return corrected_str + student_out, interpretability

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
