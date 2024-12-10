from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict
from argparse import ArgumentParser

import torch
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(Path.cwd()).parents[0]))
from data.DataHandler import DataHandler
from data.Statistics import Statistics as St
from config.baseline_config import BaselineConfig


class QATasksBaseline:
    # TODO: add documentation for the class and the methods
    """
    QATasksBaseline runs the model
    """

    def __init__(self):
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
        self.soft_accuracies_per_task: list = []

        self.accuracy: int = 0
        self.soft_accuracy: int = 0

    def load_model(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    def sample_into_parts(self, sample: Dict[str, Dict[int, str]]) -> List[List[str]]:
        """

        :param sample:
        :return:
        """

        sample_ordered = sorted(list(sample["context"].items()) + list(sample["question"].items()))
        is_question = (lambda sentence: "?" in sentence)
        parts = [[]]

        for line_id, line in sample_ordered:
            if not is_question(line) and self.to_enumerate["context"]:
                line = f"{line_id}. {line}"
            elif is_question(line) and self.to_enumerate["question"]:
                line = f"{line_id}. {line}"

            parts[-1].append(line)
            if is_question(line) and line_id != len(sample_ordered):
                parts.append([])

        return parts

    @staticmethod
    def expand_cardinal_points(abbr_news: List[str]) -> List[str]:
        cardinal_points = {
            "n": "north",
            "e": "east",
            "w": "west",
            "s": "south"
        }
        expanded_news = []
        for abbr in abbr_news:
            if abbr in cardinal_points.keys():
                expanded_news.append(cardinal_points[abbr])
            else:
                expanded_news.append(abbr)
        return expanded_news

    def iterate_task(
            self, task_id: int,
            task_data: Dict[int, Dict[str, Dict[int, str] | Dict[int, List[str]] | Dict[int, List[List[int]]]]],
            no_samples: int,
            max_new_tokens: int,
            temperature: float
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
        :param max_new_tokens: the maximum number of tokens the model will produce in its answer (cut-off)
        :param temperature: adjust the consistency of model's answers (0.1 by default)

        :return:
        """
        # results to save into the csv file
        task_results = []
        accuracies_task = []
        soft_accuracies_task = []

        # run per sample
        for sample_id, sample_data in list(task_data.items())[:no_samples]:
            expanded_answers = [self.expand_cardinal_points(ans) for ans in sample_data["answer"].values()]
            y_true_sample = [", ".join(true).lower() for true in expanded_answers]
            self.y_true.extend(y_true_sample)
            y_pred_sample = []
            sample_id_ = sample_id + 1

            # 1. Reformat the data into chunks
            sample_parts = self.sample_into_parts(sample_data)
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
                # TODO try with continue_final_message instead of add_generation_prompt
                formatted_prompt = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                print("Formatted prompt:", formatted_prompt, sep="\n", end="\n")

                # 3. Tokenize
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)
                inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}

                # 4. Generate text
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                # 5. Decode output back to string
                decoded_output = "\n".join(
                    [self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):],
                                           skip_special_tokens=True)
                     for i in range(len(outputs))]
                ).lower()
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

            soft_accuracy_sample = round(St.soft_accuracy_score(y_true_sample, y_pred_sample), 2)
            soft_accuracies_task.append(soft_accuracy_sample)
            print(f"Soft accuracy per sample {sample_id_}:", soft_accuracy_sample, end="\n\n\n")

        print("\n- TASK RESULTS -", end="\n\n")

        accuracy_task = round(sum(accuracies_task) / len(accuracies_task), 2)
        self.accuracies_per_task.append(accuracy_task)

        print(f"Accuracy score per task {task_id}:", accuracy_task)
        task_results[0]["accuracy"] = accuracy_task

        soft_accuracy_task = round(sum(soft_accuracies_task) / len(soft_accuracies_task), 2)
        self.soft_accuracies_per_task.append(soft_accuracy_task)

        print(f"Soft accuracy per task {task_id}:", soft_accuracy_task, end="\n\n")
        task_results[0]["soft_accuracy"] = soft_accuracy_task

        print(f"The work on task {task_id} is finished successfully")
        return task_results

    def run(self, cfg: BaselineConfig) -> None:
        data = DataHandler()

        results_path = cfg.repository.path + cfg.results.path + cfg.prompt.name

        if cfg.results.print_to_file:
            log_file, results_path = data.redirect_printing_to_file(path=results_path)

        data.set_results_details(results_path=results_path, headers=cfg.results.headers)

        self.set_system_prompt(prompt=cfg.prompt.text)
        self.load_model(model_name=cfg.model.name)
        print("The model is loaded successfully")

        self.total_tasks = 0
        data_in_splits = {}

        for split, to_fetch in cfg.data.splits.items():
            if to_fetch:
                data_tasks = data.read_data(path=cfg.data.path, split=split,
                                            tasks=cfg.data.task_ids)
                processed_data = data.process_data(data=data_tasks)
                self.total_tasks += len(data_tasks)
                data_in_splits[split] = processed_data

        print("The data is loaded successfully", end="\n\n")
        print("Starting to query the model", end="\n\n")

        self.to_enumerate = cfg.data.to_enumerate
        for split, tasks in data_in_splits.items():
            for task_id, task in tasks.items():
                task_result = self.iterate_task(
                    task_id=task_id, task_data=task,
                    no_samples=cfg.data.samples_per_task,
                    max_new_tokens=cfg.model.max_new_tokens,
                    temperature=cfg.model.temperature
                )
                data.save_output(data=task_result)
                print("______________________________", end="\n\n")

        print("The run is finished successfully")

        print("\n- RUN RESULTS -", end="\n\n")

        print("Processed", self.total_tasks, "tasks in total with",
              cfg.data.samples_per_task, "samples in each")
        print("Total samples processed",
              self.total_tasks * cfg.data.samples_per_task, end="\n\n")

        self.accuracy = round(accuracy_score(self.y_true, self.y_pred), 2)
        print("General accuracy:", self.accuracy)

        self.soft_accuracy = round(St.soft_accuracy_score(self.y_true, self.y_pred), 2)
        print("General soft accuracy:", self.soft_accuracy)

        row = [{"accuracy": self.accuracy,
                "soft_accuracy": self.soft_accuracy}]
        data.save_output(data=row)

        if cfg.results.print_to_file:
            # console printing must be returned
            # if printing was redirected to logs created at the beginning of the script
            # 'log_file' will exist in that case as well
            data.return_console_printing(log_file)


if __name__ == "__main__":
    # TODO: add to config if we want to print certain data
    #  (level of messages: INFO, CRITICAL and so on)

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", dest="config",
                        help="use the settings from the config file of given name "
                             "(with relative path from the config directory)",
                        metavar="CONFIG")
    args = parser.parse_args()

    cs = ConfigStore.instance()
    cs.store(name="config", node=BaselineConfig)

    with initialize(version_base=None, config_path="../config"):
        # possible TODO: to change output directory to relative:
        # ${hydra:runtime.cwd}/desired/output/directory
        if args.config:
            cfg = compose(config_name=args.config)
        else:
            # for cases of running the script interactively
            cfg = compose(config_name="baseline_config")

        baseline = QATasksBaseline()
        baseline.run(cfg)

