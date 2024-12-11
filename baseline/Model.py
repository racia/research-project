from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict

import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(Path.cwd()).parents[0]))
from data.Statistics import Statistics as St
import utils


class Baseline:
    # TODO: add documentation for the class and the methods
    """
    QATasksBaseline runs the model
    """

    def __init__(self, model_name: str, max_new_tokens: int, temperature: float):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.token = os.getenv("HUGGINGFACE")
        self.model = None
        self.tokenizer = None

        self.system_prompt_: list = []
        self.y_true, self.y_pred = [], []

        self.to_enumerate: dict = {}
        self.question_id = 0
        self.total_samples = 0
        self.total_tasks = 0

        self.accuracies_per_task: list = []
        self.soft_match_accuracies_per_task: list = []

        self.accuracy: int = 0
        self.soft_match_accuracy: int = 0

    def load_model(self) -> None:
        """
        Load the model.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def set_system_prompt(self, prompt):
        self.system_prompt_ = [{"role": "system", "content": prompt}, ]

    def get_system_prompt(self) -> List[Dict[str, str]]:
        return self.system_prompt_

    @staticmethod
    def format_prompt_part(part: list | str, role: str) -> dict:
        """

        :param part:
        :param role: only "user" or "assistant"
        :return:
        """
        if type(part) is list:
            part = "\n".join(part)
        return {"role": role, "content": part}

    def call_model(self, prompt: str) -> str:
        # 3. Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}

        # 4. Generate text
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # 5. Decode output back to string
        decoded_output = "\n".join(
            [self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):],
                                   skip_special_tokens=True)
             for i in range(len(outputs))]
        ).lower()
        return decoded_output

    def iterate_task(
            self, task_id: int,
            task_data: Dict[int, Dict[str, Dict[int, str] | Dict[int, List[str]] | Dict[int, List[List[int]]]]],
            no_samples: int,
            to_continue: bool
    ) -> List[Dict[str, int | str]]:
        """


        Parameters
        ----------
        :param task_id:
        :param task_data: task data of the following structure:
        {
            sample_id: {
               "context": {
                   line_number: line,
               },
               "question": {
                   line_number: line,
               },
               "answer": {
                   line_number: [answers],
               },
               "supporting_fact": [
                   [line_number_first_answer, ...],
               ]
            }
        }
        :param no_samples: number of samples to run per task
        :param to_continue: if we want the model to continue on the last message rather than create a separate answer

        :return:
        """
        # results to save into the csv file
        task_results = []
        accuracies_task = []
        soft_match_accuracies_task = []

        # run per sample
        for sample_id, sample_data in list(task_data.items())[:no_samples]:
            expanded_answers = [utils.expand_cardinal_points(ans) for ans in sample_data["answer"].values()]
            y_true_sample = [", ".join(true).lower() for true in expanded_answers]
            self.y_true.extend(y_true_sample)
            y_pred_sample = []
            sample_id_ = sample_id + 1

            # 1. Reformat the data into chunks
            sample_parts = utils.sample_into_parts(sample_data)
            prompt = self.get_system_prompt().copy()
            part_id = 0

            # run a conversation with one question at a time
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
                    end="\n\n\n"
                )

                # 2. Create and format the prompt
                prompt.append(self.format_prompt_part(sample_part, "user"))
                if to_continue:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        prompt, tokenize=False, continue_final_message=True
                    )
                else:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        prompt, tokenize=False, add_generation_prompt=True
                    )
                print("Formatted prompt:", formatted_prompt, sep="\n", end="\n")

                decoded_output = self.call_model(prompt=formatted_prompt)
                print("Model's output:", decoded_output, end="\n\n\n")

                # 6. Add the model's output to conversation
                prompt.append(self.format_prompt_part(decoded_output, "assistant"))
                # the model is asked question by question, so that we don't need to parse the answer
                part_result = {
                    "id": self.question_id,
                    "task_id": task_id,
                    "sample_no": sample_id_,
                    "task": "\n".join(sample_part),
                    "true_result": y_true,
                    "model_result": decoded_output,
                }

                task_results.append(part_result)
                y_pred_sample.append(decoded_output)

            self.y_pred.extend(y_pred_sample)

            print("Model's predictions for the sample:", "\t{:<18s} PREDICTED".format("GOLDEN"), sep="\n\n", end="\n\n")
            [print("\t{0:<18s} {1}".format(true, predicted.replace("\n", "\t")))
             for true, predicted in zip(y_true_sample, y_pred_sample)]
            print()

            accuracy_sample = round(accuracy_score(y_true_sample, y_pred_sample), 2)
            accuracies_task.append(accuracy_sample)
            print(f"Accuracy score per sample {sample_id_}:", accuracy_sample)

            soft_match_accuracy_sample = round(St.soft_match_accuracy_score(y_true_sample, y_pred_sample), 2)
            soft_match_accuracies_task.append(soft_match_accuracy_sample)
            print(f"Soft accuracy per sample {sample_id_}:", soft_match_accuracy_sample, end="\n\n\n")

        print("\n- TASK RESULTS -", end="\n\n")

        accuracy_task = round(sum(accuracies_task) / len(accuracies_task), 2)
        self.accuracies_per_task.append(accuracy_task)

        print(f"Accuracy score per task {task_id}:", accuracy_task)
        task_results[0]["accuracy"] = accuracy_task

        soft_match_accuracy_task = round(sum(soft_match_accuracies_task) / len(soft_match_accuracies_task), 2)
        self.soft_match_accuracies_per_task.append(soft_match_accuracy_task)

        print(f"Soft accuracy per task {task_id}:", soft_match_accuracy_task, end="\n\n")
        task_results[0]["soft_match_accuracy"] = soft_match_accuracy_task

        print(f"The work on task {task_id} is finished successfully")
        return task_results
