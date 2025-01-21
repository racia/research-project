from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.cuda.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils
from baseline.config.baseline_config import Enumerate
from data.Chat import Chat
from data.Statistics import Statistics
from prompts.Prompt import Prompt


@dataclass
class Source:
    user: str = "user"
    assistant: str = "assistant"


class Baseline:
    """
    The baseline class.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        to_enumerate: dict[Enumerate, bool],
        to_continue: bool,
        parse_output: bool,
        statistics: Statistics,
        prompt: Prompt = None,
        samples_per_task: int = 5,
    ):
        """
        Baseline class manages model runs and data flows around it.
        It is intended to use in the further settings.

        :param model_name: official name of the model to run
        :param max_new_tokens: maximum number of tokens the model would be able to answer (cut-off)
        :param temperature: the temperature of the model
        :param to_enumerate: config adding line numbers to the beginning of lines
        :param to_continue: if we want the model to continue on the last message
                            rather than create a separate answer
        :param parse_output: if we want to parse the output of the model (currently looks for 'answer' and 'reasoning')
        :param statistics: class for statistics
        :param prompt: system prompt to start conversations
        :param samples_per_task: number of samples per task for logging
        """
        self.token = os.getenv("HUGGINGFACE")

        self.model = None
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.to_continue = to_continue

        self.tokenizer = None

        self.prompt = prompt
        self.to_enumerate = to_enumerate
        self.parse_output = parse_output

        self.stats = statistics

        self.question_id = 0
        self.total_samples = 0
        self.total_tasks = 0
        self.total_parts = 0
        self.samples_per_task = samples_per_task

        self.accuracies_per_task: list = []
        self.soft_match_accuracies_per_task: list = []

        self.accuracy: int = 0
        self.soft_match_accuracy: int = 0

    def load_model(self) -> None:
        """
        Load the model and the tokenizer for the instance model name.
        """
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        torch.cuda.empty_cache()

    def call_model(self, prompt: str) -> str:
        """
        Calls that model in the following steps:
        1. tokenizes the prompt
        2. generates the answer from the inputs
        3. decodes and lowercases the answer

        :param prompt: tokenized prompt, prepared to feed into the model
        :return: response of the model
        """
        # no_grad context manager to disable gradient calculation during inference (speeds up)
        with torch.no_grad():
            # 1. Tokenize
            inputs = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            )
            inputs = {
                key: tensor.to(self.model.device) for key, tensor in inputs.items()
            }

            # autocast enables mixed precision for computational efficiency
            with autocast():
                # 2. Generate text
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                # 3. Decode output back to string
                decoded_output = "\n".join(
                    [
                        self.tokenizer.decode(
                            outputs[i][inputs["input_ids"].size(1) :],
                            skip_special_tokens=True,
                        )
                        for i in range(len(outputs))
                    ]
                ).lower()
            return decoded_output

    def iterate_task(
        self,
        task_id: int,
        task_data: dict[
            int,
            list[
                dict[
                    str,
                    dict[int, str] | dict[int, list[str]] | dict[int, list[list[int]]],
                ]
            ],
        ],
        prompt_name: str,
    ) -> list[dict[str, int | str]]:
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
        :param prompt_name: name of the prompt
        :return: results for the task in a list of dicts with each dict representing
                 one call to the model and will end up as one row of the table
        """
        task_results = []
        sample_accuracies = []
        sample_soft_match_accuracies = []

        # 1. Iterate through samples
        for sample_id, sample_parts in list(task_data.items()):
            # each sample is a new conversation
            chat = Chat(system_prompt=self.prompt.text)
            # collect the true answers
            y_true_sample = [
                ", ".join(list(part["answer"].values())[0]) for part in sample_parts
            ]
            y_pred_sample = []
            sample_id_ = sample_id + 1
            part_id = 0

            # 2. Iterate through parts (one question at a time)
            for sample_part, y_true in zip(sample_parts, y_true_sample):
                self.question_id += 1
                part_id += 1
                print(
                    "\n-* "
                    f"TASK {task_id}/{self.total_tasks} | "
                    f"SAMPLE {sample_id_}/{self.samples_per_task} | "
                    f"PART {part_id}/{len(sample_parts)} | "
                    f"{prompt_name} | "
                    f"RUN ID {self.question_id}/{self.total_parts} "
                    "*-",
                    end="\n\n\n",
                )
                # 3. Prepare the part to be in the next prompt
                formatted_part = self.prompt.format_part(
                    part=sample_part, to_enumerate=self.to_enumerate
                )
                # 4. Create and format the prompt
                chat.add_message(part=formatted_part, source="user")

                if self.to_continue:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        chat.messages, tokenize=False, continue_final_message=True
                    )
                else:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        chat.messages, tokenize=False, add_generation_prompt=True
                    )
                print(
                    "Formatted prompt:",
                    formatted_prompt,
                    sep="\n",
                    end="\n",
                )

                # 5. Call the model and yield the response
                decoded_output = self.call_model(prompt=formatted_prompt)
                print(
                    "Model's output:",
                    decoded_output,
                    end="\n\n\n",
                    flush=True,
                )

                # 6. Add the model's output to conversation
                chat.add_message(part=decoded_output, source="assistant")

                part_result = {
                    "id": self.question_id,
                    "task_id": task_id,
                    "sample_no": sample_id_,
                    "task": "\n".join(sample_part),
                    "true_result": y_true,
                    "model_result": decoded_output,
                }

                if self.parse_output:
                    parsed_output = utils.parse_output(output=decoded_output)
                    part_result["model_answer"] = parsed_output["answer"]
                    part_result["model_reasoning"] = parsed_output["reasoning"]

                task_results.append(part_result)
                y_pred_sample.append(decoded_output)

            # 7. Report the results for the sample: answers and accuracy
            print(
                "Model's predictions for the sample:",
                "\t{:<18s} PREDICTED".format("GOLDEN"),
                sep="\n\n",
                end="\n\n",
            )
            [
                print(
                    "\t{0:<18s} {1}".format(true, predicted.replace("\n", "\t")),
                )
                for true, predicted in zip(y_true_sample, y_pred_sample)
            ]

            sample_accuracy = round(
                self.stats.accuracy_score(y_true_sample, y_pred_sample), 2
            )
            sample_accuracies.append(sample_accuracy)
            print(
                f"\nAccuracy score for sample {sample_id_}:",
                sample_accuracy,
            )

            sample_soft_match_accuracy = round(
                self.stats.soft_match_accuracy_score(y_true_sample, y_pred_sample), 2
            )
            sample_soft_match_accuracies.append(sample_soft_match_accuracy)
            print(
                f"Soft accuracy for sample {sample_id_}:",
                sample_soft_match_accuracy,
                end="\n\n\n",
            )

        # 8. Report the results for the task: accuracy
        print("\n- TASK RESULTS -", end="\n\n")

        task_accuracy = round(sum(sample_accuracies) / len(sample_accuracies), 2)
        self.accuracies_per_task.append(task_accuracy)

        print(f"Accuracy score for task {task_id}:", task_accuracy)
        task_results[0]["accuracy"] = task_accuracy

        task_soft_match_accuracy = round(
            sum(sample_soft_match_accuracies) / len(sample_soft_match_accuracies), 2
        )
        self.soft_match_accuracies_per_task.append(task_soft_match_accuracy)

        print(
            f"Soft match accuracy for task {task_id}:",
            task_soft_match_accuracy,
            end="\n\n",
        )
        task_results[0]["soft_match_accuracy"] = task_soft_match_accuracy

        print(f"The work on task {task_id} is finished successfully")

        # Clear the cache
        torch.cuda.empty_cache()

        return task_results
