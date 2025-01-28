import sys
from typing import TextIO

import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline import utils
from data.Chat import Chat
from data.DataSaver import DataSaver
from data.Statistics import Statistics as St
from prompts.Prompt import Prompt


def loading(teacher_name: str, student_name: str):
    """
    Load the models and the tokenizer.
    The student model is loaded onto the first GPU, while the teacher model is loaded onto the second GPU.

    :return:teacher, student, tokenizer
    """
    student = AutoModelForCausalLM.from_pretrained(
        student_name,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        device_map={"": "cuda:1"},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    print("Successfully loaded the models and the tokenizer.", file=sys.stdout)
    return teacher, student, tokenizer


class SpeculativeDecoding:
    """
    This class handles everything related to speculative decoding.
    """

    def __init__(
        self,
        teacher: str,
        student: str,
        init_prompt: Prompt,
        eval_prompt: Prompt,
        resume_prompt: Prompt,
        teacher_max_new_tokens: int,
        student_max_new_tokens: int,
        logfile: TextIO,
    ):
        """
        Initialize the speculative decoding setting.

        The speculative decoding setting consists of a teacher model, a student model, and a tokenizer.
        The init_prompt is the initial prompt that is used to start the chain of thought of the student model.
        The eval_prompt is used to prompt the teacher to evaluate the student model's chain of thought.
        The resume_prompt is used to prompt the student model to resume the chain of thought with the corrections by
        the teacher.

        :param teacher: The teacher model
        :param student: The student model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(torch.__version__, file=sys.stdout)
        print(torch.version.cuda, file=sys.stdout)
        print(torch.cuda.is_available(), file=sys.stdout)
        print(f"Device: {self.device}", file=sys.stdout, flush=True)

        if not torch.cuda.is_available():
            raise Exception("CUDA is not available. This will be using the CPU.")

        teacher, student, tokenizer = loading(teacher, student)

        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer

        self.teacher_max_new_tokens = teacher_max_new_tokens
        self.student_max_new_tokens = student_max_new_tokens

        self.init_prompt = init_prompt
        self.eval_prompt = eval_prompt
        self.resume_prompt = resume_prompt

        self.current_task = None

        self.ds = DataSaver(log_path="log.txt")

        self.y_true, self.y_pred = [], []

        self.question_id = 0
        self.total_samples = 0
        self.total_tasks = 0

        self.accuracies_per_task: list = []
        self.soft_match_accuracies_per_task: list = []

        self.accuracy: int = 0
        self.soft_match_accuracy: int = 0

        self.log = logfile

    def generate_student(self, input_prompt: str, max_new_tokens=None) -> str:
        """
        Generate some candidates using the student model.

        The student model generates a chain of thought based on the input string.
        The maximum length of this chain of thought is handled by the max_length parameter.

        :param input_prompt: The init prompt with the input string
        :param max_new_tokens: THe maximum amount of tokens the student model should generate

        :return: The output of the student model
        """
        if max_new_tokens is None:
            max_new_tokens = self.student_max_new_tokens
        with torch.no_grad():
            encoded_in = self.tokenizer.encode(
                input_prompt, return_tensors="pt", add_special_tokens=True
            ).to("cuda:0")

            student_out = self.student.generate(
                encoded_in,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
            )

            decoded_student_out = self.tokenizer.decode(
                student_out[0], skip_special_tokens=True
            )

            return decoded_student_out

    def verify_cot(
        self, student_out, k=5, min_prob=0.05
    ) -> tuple[bool, int, str] | tuple[bool, None, None]:
        """
        Verify the candidates using the teacher model.

        Let the teacher go through the whole chain of thought proposed by the student.
        If the teacher disagrees at some point, return the teachers suggestion as the CoT step.


        :param student_out: The output of the student model
        :param k: The number of top candidates to consider from the teacher model
        :param min_prob: The minimum probability for a token in the teacher model to be considered

        :return: A tuple containing a boolean indicating whether the current CoT is valid, an integer or None indicating
        the error index and the teachers intervention or None
        """
        chain_of_thought_list = student_out.split()

        for i, token in enumerate(chain_of_thought_list):
            curr_sequence_list = chain_of_thought_list[: i + 1]
            curr_sequence_str = " ".join(curr_sequence_list)
            eval_prompt = self.eval_prompt.formulate_init_prompt(curr_sequence_str)
            input_ids = self.tokenizer.encode(eval_prompt, return_tensors="pt").to(
                "cuda:1"
            )

            with torch.no_grad():
                teacher_outputs = self.teacher(input_ids)
                teacher_logits = teacher_outputs.logits
                teacher_probs = torch.nn.functional.softmax(
                    teacher_logits[:, -1, :], dim=-1
                )

            top_probs, top_ids = torch.topk(teacher_probs, k, dim=-1)
            top_tokens_decoded = []
            for token_id in top_ids[0]:
                top_tokens_decoded.append(self.tokenizer.decode(token_id))

            if (
                token not in top_tokens_decoded
                or teacher_probs[0, self.tokenizer.encode(token)[0]] < min_prob
            ):
                suggested_token = top_tokens_decoded[0]

                return False, i, suggested_token

        return True, None, None

    def speculative_decode(self, sample_part: str, chat: Chat) -> str:
        """
        Run the speculative decoding for one instance.

        The speculative decoding consists of the following steps:
        1. Generate a chain of thought with the student model.
        2. Verify that chain of thought using the teacher model.
        3. If the teacher model disagrees, correct the chain of thought and continue with the student model.
        4. Repeat steps 2 and 3 until the teacher model agrees with the chain of thought.

        :param sample_part: The current sample
        :param chat: The current chat

        :return: The decoded output
        """
        chat.add_message(part=sample_part, role="user", model_role="student")

        init_prompt = self.tokenizer.apply_chat_template(
            chat.messages["student"], tokenize=False, add_generation_prompt=True
        )

        print(
            "Formatted prompt:",
            init_prompt,
            sep="\n",
            end="\n",
            file=self.log,
        )

        student_cot = self.generate_student(init_prompt)
        chat.add_message(part=sample_part, role="assistant", model_role="student")

        student_tokens = student_cot.split()

        is_valid, error_id, teacher_intervention = self.verify_cot(student_cot)
        chat.add_message(
            part=teacher_intervention, role="assistant", model_role="teacher"
        )

        while not is_valid:
            corrected_chain = student_tokens[:error_id] + [teacher_intervention]
            input_text = " ".join(corrected_chain)

            length_to_gen = self.student_max_new_tokens - len(corrected_chain)
            resume_prompt = self.resume_prompt.formulate_resume_prompt(input_text)
            chat.add_message(part=resume_prompt, role="user", model_role="teacher")

            new_cot = self.generate_student(resume_prompt, max_new_tokens=length_to_gen)
            new_tokens = new_cot.split()
            student_tokens = corrected_chain + new_tokens[len(corrected_chain) :]

            student_cot = " ".join(student_tokens)

            chat.add_message(part=student_cot, role="assistant", model_role="student")

            is_valid, error_id, teacher_intervention = self.verify_cot(student_cot)
            chat.add_message(
                part=teacher_intervention, role="assistant", model_role="teacher"
            )

        return student_cot

    def iterate_task(
        self,
        task_id: int,
        task_data: dict[
            int,
            dict[
                str, dict[int, str] | dict[int, list[str]] | dict[int, list[list[int]]]
            ],
        ],
        no_samples: int,
        to_enumerate: dict[enumerate, bool],
        parse_output: bool,
    ):
        """
        Run the speculative decoding setting.
        :return:
        """
        task_results = []
        accuracies_task = []
        soft_match_accuracies_task = []

        # 1. Iterate through samples
        for sample_id, sample_data in list(task_data.items())[:no_samples]:
            # each sample is a new conversation
            chat = Chat(inital_prompt=self.init_prompt.text, multi_system=True)

            # it actually gets a list of strings, not just a string
            expanded_answers = [
                utils.expand_cardinal_points(answers)
                for answers in sample_data["answer"].values()
            ]

            y_true_sample = [", ".join(true).lower() for true in expanded_answers]
            self.y_true.extend(y_true_sample)
            y_pred_sample = []
            sample_id_ = sample_id + 1

            # 2. Reformat the data into parts
            sample_parts = utils.sample_into_parts(
                sample=sample_data, to_enumerate=to_enumerate
            )

            # 3. Iterate through parts (one question at a time)
            for part_id, (sample_part, y_true) in enumerate(
                zip(sample_parts, y_true_sample), start=1
            ):
                self.question_id += 1
                print(
                    "\n-* "
                    f"TASK {task_id}/{self.total_tasks} | "
                    f"SAMPLE {sample_id_}/{no_samples} | "
                    f"PART {part_id}/{len(sample_parts)} | "
                    f"Run ID {self.question_id}"
                    " *-",
                    end="\n\n\n",
                    file=self.log,
                    flush=True,
                )

                # 4.+ 5. Call the model and yield the response
                student_cot = self.speculative_decode(sample_part, chat)

                part_result = {
                    "id": self.question_id,
                    "task_id": task_id,
                    "sample_no": sample_id_,
                    "task": "\n".join(sample_part),
                    "true_result": y_true,
                    "model_result": student_cot,
                }

                if parse_output:
                    parsed_output = utils.parse_output(output=student_cot)
                    part_result["model_answer"] = parsed_output["answer"]
                    part_result["model_reasoning"] = parsed_output["reasoning"]

                task_results.append(part_result)
                y_pred_sample.append(student_cot)

            self.y_pred.extend(y_pred_sample)

            print(f"\n Chat: {chat.messages}", file=self.log)

            # 7. Report the results for the sample: answers and accuracy
            print(
                "Model's predictions for the sample:",
                "\t{:<18s} PREDICTED".format("GOLDEN"),
                sep="\n\n",
                end="\n\n",
                file=self.log,
            )
            [
                print(
                    "\t{0:<18s} {1}".format(true, predicted.replace("\n", "\t")),
                    file=self.log,
                )
                for true, predicted in zip(y_true_sample, y_pred_sample)
            ]
            print(file=self.log)

            accuracy_sample = round(accuracy_score(y_true_sample, y_pred_sample), 2)
            accuracies_task.append(accuracy_sample)
            print(
                f"Accuracy score per sample {sample_id_}:",
                accuracy_sample,
                file=self.log,
            )

            soft_match_accuracy_sample = round(
                St.soft_match_accuracy_score(y_true_sample, y_pred_sample), 2
            )
            soft_match_accuracies_task.append(soft_match_accuracy_sample)
            print(
                f"Soft accuracy per sample {sample_id_}:",
                soft_match_accuracy_sample,
                end="\n\n\n",
                file=self.log,
            )

        # 8. Report the results for the task: accuracy
        print("\n- TASK RESULTS -", end="\n\n", file=self.log)

        accuracy_task = round(sum(accuracies_task) / len(accuracies_task), 2)
        self.accuracies_per_task.append(accuracy_task)

        print(f"Accuracy score per task {task_id}:", accuracy_task, file=self.log)
        task_results[0]["accuracy"] = accuracy_task

        soft_match_accuracy_task = round(
            sum(soft_match_accuracies_task) / len(soft_match_accuracies_task), 2
        )
        self.soft_match_accuracies_per_task.append(soft_match_accuracy_task)

        print(
            f"Soft match accuracy per task {task_id}:",
            soft_match_accuracy_task,
            end="\n\n",
            file=self.log,
        )
        task_results[0]["soft_match_accuracy"] = soft_match_accuracy_task

        print(f"The work on task {task_id} is finished successfully", file=self.log)
        return task_results
