import sys
from typing import TextIO, Tuple

import torch
from prompts.Chat import Chat
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline import utils
from data.DataSaver import DataSaver
from data.Statistics import Statistics as St
from prompts.Prompt import Prompt


def loading(teacher_name: str, student_name: str):
    """
    Load the models and the tokenizer.

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


class Feedback:
    """
    This class handles the Feedback setting.
    """

    def __init__(
        self,
        teacher: str,
        student: str,
        init_prompt: Prompt,
        feedback_prompt: Prompt,
        refine_prompt: Prompt,
        teacher_max_new_tokens: int,
        student_max_new_tokens: int,
        logfile: TextIO,
    ):
        """
        Create a feedback setting.
        In the feedback setting, the student generates a chain-of-thought and receives some feedback
        on this by the teacher. It is then asked to refine its chain-of-thought based on this feedback.

        :param teacher: the Llama teacher model
        :param student: the Llama student model
        :param init_prompt: the initial prompt for the student
        :param feedback_prompt: the prompt for the teacher to provide feedback
        :param refine_prompt: the prompt for the student to refine its chain-of-thought
        :param teacher_max_new_tokens: int, the maximum amount of new tokens the teacher should generate
        :param student_max_new_tokens: int, the maximum amount of new tokens the student should generate
        :param: logfile: the file the results should be saved to

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
        self.feedback_prompt = feedback_prompt
        self.refine_prompt = refine_prompt

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

    def check_feedback(self, feedback: str) -> bool:
        """
        Check whether the teacher suggests changes in the feedback or the teacher is
        satisfied with the students current chain of thought.

        :param feedback: the feedback given by the teacher

        :return: bool, whether the teacher is satisfied with the students current chain of thought
        """
        if "xyz" in feedback:
            return True
        else:
            return False

    def generate_student(self, input_prompt: str, student_max_new_tokens=None) -> str:
        """
        Generate some candidates using the student model.

        The student model generates a chain of thought based on the input string.
        The maximum length of this chain of thought is handled by the max_length parameter.

        :param input_prompt: The init prompt with the input string

        :return: The output of the student model
        """
        if student_max_new_tokens is None:
            student_max_new_tokens = self.student_max_new_tokens

        encoded_in = self.tokenizer.encode(
            input_prompt, return_tensors="pt", add_special_tokens=True
        ).to("cuda:0")

        student_out = self.student.generate(
            encoded_in,
            max_new_tokens=student_max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        decoded_student_out = self.tokenizer.decode(
            student_out[0], skip_special_tokens=True
        )
        return decoded_student_out

    def give_feedback(self, student_out: str) -> Tuple[str, bool]:
        """
        Prompt the teacher to give feedback on the current chain of thought of the student.

        :param student_out: output generated by the student
        :return: (str, bool), the feedback given by the teacher and whether the teacher is satisfied
        """
        feedback_prompt = self.feedback_prompt.formulate_feedback_prompt(student_out)

        input_ids = self.tokenizer.encode(feedback_prompt, return_tensors="pt").to(
            "cuda:1"
        )

        teacher_feedback = self.teacher.generate(input_ids)

        decoded_teacher_feedback = self.tokenizer.decode(
            teacher_feedback[0], skip_special_tokens=True
        )

        is_valid = self.check_feedback(decoded_teacher_feedback)
        self.ds.save_teacher_feedback(
            prompt=feedback_prompt, feedback=decoded_teacher_feedback, is_valid=is_valid
        )

        return decoded_teacher_feedback, is_valid

    def refine(self, teacher_feedback: str) -> str:
        """
        Prompt the student to refine its chain of thought according to the teacher it received from
        the teacher.

        :param teacher_feedback: the feedback generated by the teacher

        :return: str, the refined chain of thought
        """
        refine_prompt = self.refine_prompt.formulate_refine_prompt(teacher_feedback)

        input_ids = self.tokenizer.encode(refine_prompt, return_tensors="pt").to(
            "cuda:0"
        )

        refined_cot = self.student.generate(input_ids)

        decoded_refined_cot = self.tokenizer.decode(
            refined_cot[0], skip_special_tokens=True
        )

        return decoded_refined_cot

    def feedback_setting(self, task_sample: str, max_new_tokens: int):
        """
        Run the feedback setting.
        The feedback setting consists of the following steps:
        1. The student generates a chain of thought,
        2. The teacher gives feedback on this chain of thought.
        3. The student takes in the feedback of the teacher and refines its chain of thought.
        4. Step 2 and 3 are repeated until no further feedback is provided by the teacher.

        :return: str, the final chain of thought
        """
        student_cot = self.generate_student(
            input_prompt=task_sample, student_max_new_tokens=max_new_tokens
        )

        feedback, is_valid = self.give_feedback(student_cot)

        while not is_valid:
            student_cot = self.refine(
                feedback
            )  # Update the refined CoT in `student_cot`
            feedback, is_valid = self.give_feedback(student_cot)

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

            part_id = 0

            # 3. Iterate through parts (one question at a time)
            for sample_part, y_true in zip(sample_parts, y_true_sample):
                self.question_id += 1
                part_id += 1
                print(
                    "\n-* "
                    f"TASK {task_id}/{self.total_tasks} | "
                    f"SAMPLE {sample_id_}/{no_samples} | "
                    f"PART {part_id}/{len(sample_parts)} | "
                    f"Run ID {self.question_id}"
                    " *-",
                    end="\n\n\n",
                    file=self.log,
                )

                # 4.+ 5. Call the model and yield the response
                student_cot = self.feedback_setting(sample_part, chat)

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
