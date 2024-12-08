import json
import os
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Union

import sys

sys.path.insert(0, str(Path(Path.cwd()).parents[0]))
from data.DataHandler import DataHandler


class QATasksBaseline:
    # TODO: add documentation for the class and the methods
    """

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
    def soft_accuracy_score(true_values, predicted_values):
        true_in_predicted = 0
        for true, prediction in zip(true_values, predicted_values):
            if true.lower() in prediction.lower():
                true_in_predicted += 1
            # for partial answer of questions with two supporting facts
            elif prediction.lower() in true.lower():
                true_in_predicted += 0.5
        if true_in_predicted == 0:
            return 0
        return true_in_predicted / len(true_values)

    # TODO: generalize the three functions into one (make_data with possible parameters: train, eval, test)
    # def get_test_files(self) -> Dict[int, Dict[str, List[str]]]:
    #     """
    #     {task_id: {train/test: [file_names]}}
    #     :return:
    #     """
    #     test_data = {}
    #     for r, d, f in os.walk(self.data_dir):
    #         #        print(r,d,f)
    #         for task_id in range(1, 21):
    #             pattern_test = re.compile(f"qa{task_id}_[\\w-]+_test\\.txt")
    #             test = list(filter(lambda x: re.findall(pattern_test, x), f))
    #             print(f"Loaded Test Data: {test[0]}\n")
    #             test_data[task_id] = {}
    #             test_data[task_id]["test"] = test[0]
    #     return test_data

    # def get_valid_files(self):
    #     pass

    # def get_train_files(self) -> Dict[int, Dict[str, List[str]]]:
    #     train_data = {}
    #     for r, d, f in os.walk(self.data_dir):
    #         #        print(r,d,f)
    #         for i in range(1, 21):
    #             pattern_train = re.compile(f"qa{i}_[\\w-]+_train\\.txt")
    #             train = list(filter(lambda x: re.findall(pattern_train, x), f))
    #             print(f"Loaded Train Data: {train[0]}\n")
    #             train_data[i] = {}
    #             train_data[i]["train"] = train[0]
    #     return train_data

    # turns a task into samples?
    # TODO: add simple data examples to each function (what goes in and out + data types in the definitions)
    # def make_task_data(self, data: dict, task_id: int) -> List[Dict[int, str]]:
    #     """
    #     Adds samples of a task to the all the task data as a list of dicts of one sequence of context sentences
    #
    #     :param data:
    #     :param task_id:
    #     :return:
    #     """
    #     # as far as I understood, "test" should be a parameter
    #     self.task_data.clear()
    #     """{task_id: {train|eval|test: [file_names]}}"""
    #     path = self.data_dir + data[task_id]["test"]
    #     with open(path, "r", encoding="utf-8") as f:
    #         lines = f.readlines()
    #         # self.task_data.append({})
    #         for i, line in enumerate(lines):
    #             line = line.strip()  # Remove newlines
    #             sentence_id, sentence = line.split(maxsplit=1)
    #             sentence_is_question = "\t" in sentence
    #             if sentence_id == "1":  # and i != (len(lines) - 1)
    #                 self.task_data.append({})
    #                 self.task_data[-1][0] = []
    #             if sentence_is_question:
    #                 question_line = sentence.split("\t")
    #                 self.task_data[-1][int(sentence_id)] = question_line.pop(0)  # add the question to the sample
    #                 # sentence ids start from 1, so we can reserve 0 for question answers
    #                 self.task_data[-1][0].append(question_line)  # add the answer and references with a separate key 0
    #             else:
    #                 self.task_data[-1][int(sentence_id)] = sentence
    #     return self.task_data

    # def format_sample(self, sample_inx: int, to_enumerate: bool = True) -> List[Dict[str, str]]:
    #     """
    #
    #     Unused value - references
    #
    #     :param sample_inx:
    #     :param to_enumerate:
    #     :return: a dict with "role" - "content" with one entry of context sentence sequence
    #     """
    #     sample = self.task_data[sample_inx]  # context sentences and questions
    #     references = sample.pop(0)  # Removes the list with answers and references from the sample dict
    #     # Separates the sample answers from the references
    #     answers = [reference_line.pop(0) for reference_line in references]
    #     # 'references' now contains only refs
    #
    #     if to_enumerate:
    #         sample = "\n".join([". ".join((str(i), sentence)) for i, sentence in list(sample.items())])
    #     else:
    #         sample = list(sample.values())  # Strings of context and the question
    #         sample = "\n".join(sample)
    #
    #     sample_parts = [{"role": "user", "content": part + "?"} for part in sample.split("?") if part.strip()]
    #
    #     print("Formatted sample in parts:\n", json.dumps(sample_parts, indent=2), "\n")
    #     self.y_true.append(answers)  # Add the answer to y_true (without the references)
    #     return sample_parts

    @staticmethod
    def format_prompt_part(part: Union[list, str], role: str) -> dict:
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

        # sample_len = len(sample["context"]) + len(sample["question"])
        # counter = 0
        #
        # for line_id, context in sample["context"].items():
        #     counter += 1
        #
        #     # if a question needs to be inserted in the middle
        #     if counter != line_id:
        #         parts[-1].append(f'{counter}. {sample["question"][counter]}'
        #                          if self.to_enumerate
        #                          else sample["question"][counter])
        #         # start new part
        #         parts.append([])
        #         counter += 1
        #
        #     if self.to_enumerate:
        #         context = f"{line_id}. {context}"
        #     parts[-1].append(context)
        #
        #     # add the last question
        #     if line_id + 1 == sample_len:
        #         parts[-1].append(f'{line_id + 1}. {sample["question"][line_id + 1]}'
        #                          if self.to_enumerate
        #                          else sample["question"][line_id + 1])
        #
        # return parts

    @staticmethod
    def expand_cardinal_points(abbr_news: List[str]) -> List[str]:
        expanded_news = []
        for abbr in abbr_news:
            match abbr:
                case "n":
                    expanded_news.append("north")
                case "e":
                    expanded_news.append("east")
                case "w":
                    expanded_news.append("west")
                case "s":
                    expanded_news.append("south")
                case _:
                    expanded_news.append(abbr)
        return expanded_news

    def iterate_task(
            self, task_id: int,
            task_data: Dict[int, Dict[str, Dict[int, str] | Dict[int, List[str]] | Dict[int, List[List[int]]]]],
            no_samples: int = -1
    ) -> list:
        """


        Parameters
        ----------
        :param task_id:
        :param task_data:
                {
                    sample_id: {
                        "context": {
                            line_number: line,
                            line_number: line, ...}
                        "question": {
                            line_number: line,
                            line_number: line, ...}
                        "answer": {
                            line_number: [answers],
                            line_number: [answers], ...}
                        "supporting_fact": [[line_number_first_answer, ...],
                                            [line_number_second_answer, ...],
                                            ...]
                        }
                }
        :param no_samples: number of samples to run per task

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
                    max_new_tokens=12,
                    temperature=0.1,
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
                y_pred_sample.append(decoded_output)

                part_result = {
                    "id": self.question_id,
                    "task_id": task_id,
                    "sample_no": sample_id_,
                    "task": "\n".join(sample_part),
                    "true_result": y_true,
                    "model_result": decoded_output,
                }
                # part_result = [self.question_id, task_id, sample_id,
                #                "\n".join(sample_part), y_true, decoded_output,]
                task_results.append(part_result)

            # y_pred_task.extend(y_pred_sample)
            self.y_pred.extend(y_pred_sample)

            print("Model's predictions for the sample:", "\t{:<18s} PREDICTED".format("GOLDEN"), sep="\n\n", end="\n\n")
            [print("\t{0:<18s} {1}".format(true, predicted.replace("\n", "\t")))
             for true, predicted in zip(y_true_sample, y_pred_sample)]
            print()

            accuracy_sample = accuracy_score(y_true_sample, y_pred_sample)
            accuracies_task.append(accuracy_sample)
            print(f"Accuracy score per sample {sample_id_}:", accuracy_sample)

            soft_accuracy_sample = self.soft_accuracy_score(y_true_sample, y_pred_sample)
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

    # def iterate_context(self, task_id):
    #     """
    #
    #     :param task_id:
    #     :return:
    #     """
    #     task_results = []
    #     accuracy_task = []
    #     in_accuracy_task = []
    #     samples_per_task = 5
    #     for sample_inx in range(samples_per_task):
    #         self.sample_id += 1
    #         sample_results = []
    #         total_tasks = samples_per_task * 20
    #         print(f"\n\n-* TASK {task_id} | SAMPLE {sample_inx+1} | ID {self.sample_id}/{total_tasks} *-\n\n")
    #         # 1. Add sample
    #         sample_parts = self.format_sample(sample_inx, to_enumerate=False)
    #         """sample parts = [{"role": "user", "content": part+"?"}, {"role": "user", "content": part+"?"}]"""
    #         [sample_results.append([task_id, sample_inx+1, sample_part["content"]]) for sample_part in sample_parts]
    #         """sample_prompt = [system_prompt,
    #                             {"role": "user", "content": part+"?"},
    #                             {"role": "assistant", "content": answer}]"""
    #         sample_prompt = self.get_system_prompt().copy()
    #
    #         self.y_pred.append([])
    #         for sample_part in sample_parts:
    #             # 2. Create generation prompt
    #             sample_prompt += [sample_part]
    #             formatted_prompt = self.tokenizer.apply_chat_template(
    #                 sample_prompt, tokenize=False, add_generation_prompt=True
    #             )
    #             print("Formatted prompt:", formatted_prompt, sep="\n", end="\n")
    #             # 3. Tokenize
    #             inputs = self.tokenizer(
    #                 formatted_prompt, return_tensors="pt", add_special_tokens=True
    #             )
    #             inputs = {
    #                 key: tensor.to(self.model.device) for key, tensor in inputs.items()
    #             }
    #             # 4. Generate text
    #             outputs = self.model.generate(
    #                 **inputs,
    #                 max_new_tokens=12,
    #                 temperature=0.1,
    #                 pad_token_id=self.tokenizer.eos_token_id,
    #             )
    #
    #             # 5. Decode output back to string
    #             # decoded_output = self.tokenizer.decode(
    #             #     outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True
    #             # )
    #             decoded_output = "\n".join([self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):],
    #                                                               skip_special_tokens=True)
    #                                         for i in range(len(outputs))])
    #             # ans = response[0]["generated_text"][-1]["content"] # Obtain answer
    #             # the model is asked question by question, so that we don't need to parse the answer
    #             self.y_pred[-1].append(decoded_output.lower())
    #             print("Model's output:", decoded_output, end="\n\n\n")  # Print qa
    #             # 6. Add the model's output for a sample part to conversation
    #             sample_prompt += [{"role": "assistant", "content": decoded_output}]
    #
    #         [result.extend([true, pred]) for result, true, pred in
    #          zip(sample_results, self.y_true[-1], self.y_pred[-1])]
    #         task_results.extend(sample_results)
    #
    #         accuracy_sample = accuracy_score(self.y_true[-1], self.y_pred[-1])
    #         accuracy_task.append(accuracy_sample)
    #         self.all_accuracies.append(accuracy_sample)
    #         print(f"Accuracy score for sample {sample_inx+1}:", round(accuracy_sample, 2))
    #
    #         in_accuracy_sample = self.in_accuracy_score(self.y_true[-1], self.y_pred[-1])
    #         in_accuracy_task.append(in_accuracy_sample)
    #         self.all_in_accuracies.append(accuracy_sample)
    #         print(f"The model's output containing the correct one:", round(in_accuracy_sample, 2))
    #
    #         print("______________________________")
    #
    #     print("\n\nModel's predictions for the sample:", "\tGOLDEN\t\tPREDICTED\n", sep="\n\n")
    #     y_pred_joined = [[prediction.replace("\n", " ") for prediction in part_predictions]
    #                      for part_predictions in self.y_pred]
    #     [print(f"\t{golden}\t\t{prediction}") for golden, prediction in zip(self.y_true[-1], y_pred_joined[-1])]
    #
    #     accuracy = round(sum(accuracy_task) / len(accuracy_task), 2)
    #     print(f"Accuracy score per task {task_id}:", accuracy)
    #     # print("General accuracy:", round(sum(self.all_accuracies) / len(self.all_accuracies), 2))
    #
    #     in_accuracy = round(sum(in_accuracy_task) / len(in_accuracy_task), 2)
    #     print(f"The model's output containing the correct one:", in_accuracy)
    #     # print("Generally, the model's output contains the correct one:",
    #     # round(sum(self.all_in_accuracies) / len(self.all_in_accuracies), 2))
    #
    #     print(f"The work on task {task_num} is finished successfully")
    #
    #     # [result.extend([true, pred]) for true, pred, result in zip(self.y_true[-1], self.y_pred[-1], task_results)]
    #     with open("results/prompt_11_.csv", "a+", encoding="utf-8") as f:
    #         writer = csv.writer(f, delimiter="\t")
    #         task_results[0].extend([accuracy, in_accuracy])
    #         writer.writerows(task_results)
    #
    #     print(f"The results of task {task_num} are saved successfully")


if __name__ == "__main__":
    # TODO?: move prompt into files
    # TODO: add to config if we want to print certain data (level of messages: INFO, CRITICAL and so on)

    # config dict
    conf = {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "home_dir": str(Path(Path.cwd()).parents[1]),  # for data to be stored next to the research project
        "data_path": "/tasks_1-20_v1-2/en-valid/",
        "prompt_num": "prompt_0_",
        "prompt_text": """You will be given a sequence of context sentences with questions included. \
Please answer each such question by only providing the correct answer. \
For example: The correct answer to: John is on the playground. Mary is in the kitchen. Where is John? \
is: Playground""",
        "results_path": "results/",
        # into variables?
        "headers": [
            "id",
            "task_id",
            "sample_no",
            "task",
            "true_result",
            "model_result",
            "accuracy",
            "soft_accuracy",
        ],
        "run_splits": {"train": False, "valid": True, "test": False},
        "task_ids": [19],  # list(range(1, 21)),  # List[str]|None to get all
        "samples_per_task": 5,
        # if to add line numbers to the sentences in the prompt
        "to_enumerate": {"context": True, "question": False},

    }

    baseline = QATasksBaseline()
    baseline.set_system_prompt(conf["prompt_text"])
    baseline.load_model(model_name=conf["model_name"])
    print("The model is loaded successfully")

    data = DataHandler()

    baseline.total_tasks = 0
    data_in_splits = {}

    for split, to_fetch in conf["run_splits"].items():
        if to_fetch:
            data_tasks = data.read_data(
                conf["home_dir"] + conf["data_path"], split=split, tasks=conf["task_ids"]
            )
            processed_data = data.process_data(data_tasks)
            baseline.total_tasks += len(data_tasks)
            data_in_splits[split] = processed_data

    print("The data is loaded successfully", end="\n\n")
    print("Starting to query the model", end="\n\n")

    baseline.to_enumerate = conf["to_enumerate"]
    for split, tasks in data_in_splits.items():
        for task_id, task in tasks.items():
            task_result = baseline.iterate_task(task_id=task_id, task_data=task,
                                                no_samples=conf["samples_per_task"])
            data.save_output(
                path=conf["results_path"] + conf["prompt_num"],
                headers=conf["headers"], data=task_result
            )
            print("______________________________", end="\n\n")

    # for task_num in conf["task_ids"]:
    #     # qa_task_iterate.make_task_data(data_splits["valid"], task_num)
    #     task_results = qa_task_baseline.iterate_context(task_num)
    #     data.save_output(
    #         path=conf["results_path"] + conf["prompt"],
    #         headers=conf["headers"], data=task_results
    #     )
    #     print("______________________________", end="\n\n")

    print("The run is finished successfully")

    print("\n- RUN RESULTS -", end="\n\n")

    print("Processed", baseline.total_tasks, "tasks in total with", conf["samples_per_task"], "samples in each")
    print("Total samples processed", baseline.total_tasks * conf["samples_per_task"], end="\n\n")

    baseline.accuracy = accuracy_score(baseline.y_true, baseline.y_pred)
    print("General accuracy:", baseline.accuracy)

    baseline.soft_accuracy = baseline.soft_accuracy_score(baseline.y_true, baseline.y_pred)
    print("General soft accuracy:", baseline.soft_accuracy)

    # row = [""] * (len(conf["headers"]) - 2) + [qa_task_baseline.accuracy, qa_task_baseline.soft_accuracy]
    row = [{"accuracy": baseline.accuracy,
            "soft_accuracy": baseline.soft_accuracy}]
    data.save_output(
        path=conf["results_path"] + conf["prompt_num"],
        headers=conf["headers"], data=row
    )
