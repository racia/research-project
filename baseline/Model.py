from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Union, TextIO

import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline.config.baseline_config import Enumerate
from data.Statistics import Statistics as St
import utils


@dataclass
class Role:
    user = "user"
    assistant = "assistant"


class Baseline:
    def __init__(self, model_name: str, max_new_tokens: int,
                 temperature: float, log_file: TextIO):
        """
        Baseline class manages model runs and data flows around it.
        It is intended to use in the further settings.

        :param model_name: official name of the model to run
        :param max_new_tokens: maximum number of tokens the model would be able
                               to answer (cut-off)
        :param temperature: the temperature of the model
        :param log_file: 'sys.stdout' or a log file (if printing was redirected)
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.log = log_file

        self.token = os.getenv("HUGGINGFACE")
        self.model = None
        self.tokenizer = None

        self.system_prompt_: list = []
        self.y_true, self.y_pred = [], []

        self.question_id = 0
        self.total_samples = 0
        self.total_tasks = 0

        self.accuracies_per_task: list = []
        self.soft_match_accuracies_per_task: list = []

        self.accuracy: int = 0
        self.soft_match_accuracy: int = 0

    def load_model(self) -> None:
        """Load the model and the tokenizer for the instance model name."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def set_system_prompt(self, prompt):
        """Set the system prompt in a formatted way."""
        self.system_prompt_ = [{"role": "system", "content": prompt},]

    def get_system_prompt(self) -> List[Dict[str, str]]:
        """Get the system prompt."""
        return self.system_prompt_

    @staticmethod
    def format_prompt_part(part: str | List[str], role: Union[Role.user, Role.assistant]) \
            -> Dict[str, str]:
        """
        Formats the prompt by managing the data type and putting in into
        a dictionary the model expects.

        :param part: part of a sample as a string or a list of strings
        :param role: the producer of the message
        :return: prompt formatted as a dict
        """
        if type(part) is list:
            part = "\n".join(part)
        return {"role": role, "content": part}

    def call_model(self, prompt: str) -> str:
        """
        Calls that model in the following steps:
        1. tokenizes the prompt
        2. generates the answer from the inputs
        3. decodes and lowercases the answer

        :param prompt: tokenized prompt, prepared to feed into the model
        :return: response of the model
        """
        # 1. Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}

        # 2. Generate text
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # 3. Decode output back to string
        decoded_output = "\n".join(
            [self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):],
                                   skip_special_tokens=True)
             for i in range(len(outputs))]
        ).lower()
        return decoded_output

    def iterate_task(
            self, task_id: int,
            task_data:
            Dict[int, Dict[str, Dict[int, str] | Dict[int, List[str]] | Dict[int, List[List[int]]]]],
            no_samples: int,
            to_enumerate: dict[Enumerate, bool],
            to_continue: bool
    ) -> List[Dict[str, int | str]]:
        """
        Manages the data flow in and out of the model, iteratively going through 
        each part of each sample in a given task, recoding the results to report.
        
        The following steps apply:
        1. iterate through tasks
        2. reformat the data into parts
           Hint: a part is defined as a list with context sentences finished with 
                 a question.
        3. iterate through parts
        4. create and format the prompt
        5. call the model and yield the response
        6. add the model's output to conversation
        7. report the results for a sample: answers and accuracy
        8. report the results for the task:  accuracy
        
        :param task_id: task id corresponding to task name in original dataset
        :param task_data: task data of the following structure:
        {
            sample_id: {
               "context": {
                   line_number: line,
                   ...
               },
               "question": {
                   line_number: line,
                   ...
               },
               "answer": {
                   line_number: [answer, ...],
               },
               "supporting_fact": [
                   [line_number_first_answer, ...],
               ]
            }
        }
        :param no_samples: number of samples to run per task
        :param to_enumerate: config adding line numbers to the beginning of lines
        :param to_continue: if we want the model to continue on the last message 
                            rather than create a separate answer
        :return: results for the task in a list of dicts with each dict representing 
                 one call to the model and will end up as one row of the table
        """
        task_results = []
        accuracies_task = []
        soft_match_accuracies_task = []

        # 1. Iterate through samples
        for sample_id, sample_data in list(task_data.items())[:no_samples]:
            # it actually gets a list of strings, not just a string
            expanded_answers = [utils.expand_cardinal_points(answers)
                                for answers in sample_data["answer"].values()]
            y_true_sample = [", ".join(true).lower() for true in expanded_answers]
            self.y_true.extend(y_true_sample)
            y_pred_sample = []
            sample_id_ = sample_id + 1

            # 2. Reformat the data into parts
            sample_parts = utils.sample_into_parts(sample=sample_data,
                                                   to_enumerate=to_enumerate)
            prompt = self.get_system_prompt().copy()
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
                    end="\n\n\n", file=self.log
                )

                # 4. Create and format the prompt
                prompt.append(self.format_prompt_part(part=sample_part,
                                                      role="user"))
                if to_continue:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        prompt, tokenize=False, continue_final_message=True
                    )
                else:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        prompt, tokenize=False, add_generation_prompt=True
                    )
                print("Formatted prompt:", formatted_prompt, sep="\n",
                      end="\n", file=self.log)

                # 5. Call the model and yield the response
                decoded_output = self.call_model(prompt=formatted_prompt)
                print("Model's output:", decoded_output, end="\n\n\n",
                      file=self.log)

                # 6. Add the model's output to conversation
                prompt.append(self.format_prompt_part(part=decoded_output,
                                                      role="assistant"))
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

            # 7. Report the results for the sample: answers and accuracy
            print("Model's predictions for the sample:",
                  "\t{:<18s} PREDICTED".format("GOLDEN"),
                  sep="\n\n", end="\n\n", file=self.log)
            [print("\t{0:<18s} {1}".format(true,
                                           predicted.replace("\n", "\t")), file=self.log)
             for true, predicted in zip(y_true_sample, y_pred_sample)]
            print(file=self.log)

            accuracy_sample = round(accuracy_score(y_true_sample, y_pred_sample), 2)
            accuracies_task.append(accuracy_sample)
            print(f"Accuracy score per sample {sample_id_}:", accuracy_sample, file=self.log)

            soft_match_accuracy_sample = round(St.soft_match_accuracy_score(y_true_sample,
                                                                            y_pred_sample), 2)
            soft_match_accuracies_task.append(soft_match_accuracy_sample)
            print(f"Soft accuracy per sample {sample_id_}:", soft_match_accuracy_sample,
                  end="\n\n\n", file=self.log)

        # 8. Report the results for the task: accuracy
        print("\n- TASK RESULTS -", end="\n\n", file=self.log)

        accuracy_task = round(sum(accuracies_task) / len(accuracies_task), 2)
        self.accuracies_per_task.append(accuracy_task)

        print(f"Accuracy score per task {task_id}:", accuracy_task, file=self.log)
        task_results[0]["accuracy"] = accuracy_task

        soft_match_accuracy_task = round(sum(soft_match_accuracies_task) /
                                         len(soft_match_accuracies_task), 2)
        self.soft_match_accuracies_per_task.append(soft_match_accuracy_task)

        print(f"Soft match accuracy per task {task_id}:", soft_match_accuracy_task,
              end="\n\n", file=self.log)
        task_results[0]["soft_match_accuracy"] = soft_match_accuracy_task

        print(f"The work on task {task_id} is finished successfully", file=self.log)
        return task_results
