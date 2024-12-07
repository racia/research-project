import json
import os
import re
from pathlib import Path
import csv
import torch
from sklearn.metrics import accuracy_score
# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

class QATaskIterate:
    # TODO: add documentation for the class and the methods
    """
    """
    def __init__(self):
        self.home_dir = Path.home()
        # TODO into config
        self.data_dir = f"{self.home_dir}/tasks_1-20_v1-2/en/"
        self.token = os.getenv("HUGGINGFACE")
        # TODO into config
        # Use the appropriate model name from the Hugging Face Hub
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
        # TODO into config: prompt
        self.system_prompt_ = [{"role": "system", "content": """You will be given a sequence of context sentences and \
then asked questions about them. Please answer each such question to your best abilities. Example:
Context sentences:
John is on the playground.
Mary is in the kitchen.
Question: Where is John?
Correct answer: Playground"""},]
        self.y_true, self.y_pred = [], []
        self.task_data = []
        self.sample_id = 0
        self.all_accuracies = []
        self.all_in_accuracies = []
    def get_system_prompt(self) -> List[Dict[str, str]]:
        return self.system_prompt_
    # TODO: generalize the three functions into one (make_data with possible parameters: train, eval, test)
    def make_test_data(self) -> Dict[int, Dict[str, List[str]]]:
        """
        {task_id: {train/test: [file_names]}}
        :return:
        """
        test_data = {}
        for r, d, f in os.walk(self.data_dir):
            #        print(r,d,f)
            for task_id in range(1, 21):
                pattern_test = re.compile(f"qa{task_id}_[\\w-]+_test\\.txt")
                test = list(filter(lambda x: re.findall(pattern_test, x), f))
                print(f"Loaded Test Data: {test[0]}\n")
                test_data[task_id] = {}
                test_data[task_id]["test"] = test[0]
        return test_data
    def make_eval_data(self):
        pass
    def make_train_data(self) -> Dict[int, Dict[str, List[str]]]:
        train_data = {}
        for r, d, f in os.walk(self.data_dir):
            #        print(r,d,f)
            for i in range(1, 21):
                pattern_train = re.compile(f"qa{i}_[\\w-]+_train\\.txt")
                train = list(filter(lambda x: re.findall(pattern_train, x), f))
                print(f"Loaded Train Data: {train[0]}\n")
                train_data[i] = {}
                train_data[i]["train"] = train[0]
        return train_data
    # turns a task into samples?
    # TODO: add simple data examples to each function (what goes in and out + data types in the definitions)
    def make_task_data(self, data: dict, task_id: int) -> List[Dict[int, str]]:
        """
        Adds samples of a task to the all the task data as a list of dicts of one sequence of context sentences
        :param data:
        :param task_id:
        :return:
        """
        # as far as I understood, "test" should be a parameter
        self.task_data.clear()
        """{task_id: {train|eval|test: [file_names]}}"""
        path = self.data_dir + data[task_id]["test"]
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # self.task_data.append({})
            for i, line in enumerate(lines):
                line = line.strip()  # Remove newlines
                sentence_id, sentence = line.split(maxsplit=1)
                sentence_is_question = "\t" in sentence
                if sentence_id == "1":  # and i != (len(lines) - 1)
                    self.task_data.append({})
                    self.task_data[-1][0] = []
                if sentence_is_question:
                    question_line = sentence.split("\t")
                    self.task_data[-1][int(sentence_id)] = question_line.pop(0)  # add the question to the sample
                    # sentence ids start from 1, so we can reserve 0 for question answers
                    self.task_data[-1][0].append(question_line)  # add the answer and references with a separate key 0
                else:
                    self.task_data[-1][int(sentence_id)] = sentence
        return self.task_data
    def format_sample(self, sample_inx: int, to_enumerate: bool = True) -> List[Dict[str, str]]:
        """
        Unused value - references
        :param sample_inx:
        :param to_enumerate:
        :return: a dict with "role" - "content" with one entry of context sentence sequence
        """
        sample = self.task_data[sample_inx]  # context sentences and questions
        references = sample.pop(0)  # Removes the list with answers and references from the sample dict
        # Separates the sample answers from the references
        answers = [reference_line.pop(0) for reference_line in references]
        # 'references' now contains only refs
        if to_enumerate:
            sample = "\n".join([". ".join((str(i), sentence)) for i, sentence in list(sample.items())])
        else:
            sample = list(sample.values())  # Strings of context and the question
            sample = "\n".join(sample)
        sample_parts = [{"role": "user", "content": part+"?"} for part in sample.split("?") if part.strip()]
        print("Formatted sample in parts:\n", json.dumps(sample_parts, indent=2), "\n")
        self.y_true.append(answers)  # Add the answer to y_true (without the references)
        return sample_parts
    @staticmethod
    def in_accuracy_score(true_values, predicted_values):
        true_in_predicted = 0
        for true, prediction in zip(true_values, predicted_values):
            if true in prediction:
                true_in_predicted += 1
        return true_in_predicted / len(true_values)
    def iterate_context(self, task_id):
        """
        :param task_id:
        :return:
        """
        task_results = []
        accuracy_task = []
        in_accuracy_task = []
        # TODO into config: sample
        samples_per_task = 5
        for sample_inx in range(samples_per_task):
            self.sample_id += 1
            sample_results = []
            # TODO into config: number of tasks
            total_tasks = samples_per_task * 20
            print(f"\n\n-* TASK {task_id} | SAMPLE {sample_inx+1} | ID {self.sample_id}/{total_tasks} *-\n\n")
            # 1. Add sample
            sample_parts = self.format_sample(sample_inx, to_enumerate=False)
            """sample parts = [{"role": "user", "content": part+"?"}, {"role": "user", "content": part+"?"}]"""
            # TODO move into format_sample
            [sample_results.append([task_id, sample_inx+1, sample_part["content"]]) for sample_part in sample_parts]
            """sample_prompt = [system_prompt, 
                                {"role": "user", "content": part+"?"},
                                {"role": "assistant", "content": answer}]"""
            sample_prompt = self.get_system_prompt().copy()
            self.y_pred.append([])
            for sample_part in sample_parts:
                # 2. Create generation prompt
                sample_prompt += [sample_part]
                # TODO try with continue_final_message instead of add_generation_prompt
                formatted_prompt = self.tokenizer.apply_chat_template(
                    sample_prompt, tokenize=False, add_generation_prompt=True
                )
                print("Formatted prompt:", formatted_prompt, sep="\n", end="\n")
                # 3. Tokenize
                inputs = self.tokenizer(
                    formatted_prompt, return_tensors="pt", add_special_tokens=True
                )
                inputs = {
                    key: tensor.to(self.model.device) for key, tensor in inputs.items()
                }
                # 4. Generate text
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=12,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                # 5. Decode output back to string
                # decoded_output = self.tokenizer.decode(
                #     outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True
                # )
                decoded_output = "\n".join([self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):],
                                                                  skip_special_tokens=True)
                                            for i in range(len(outputs))])
                # ans = response[0]["generated_text"][-1]["content"] # Obtain answer
                # the model is asked question by question, so that we don't need to parse the answer
                self.y_pred[-1].append(decoded_output.lower())
                print("Model's output:", decoded_output, end="\n\n\n")  # Print qa
                # 6. Add the model's output for a sample part to conversation
                sample_prompt += [{"role": "assistant", "content": decoded_output}]
            [result.extend([true, pred]) for result, true, pred in
             zip(sample_results, self.y_true[-1], self.y_pred[-1])]
            task_results.extend(sample_results)
            accuracy_sample = accuracy_score(self.y_true[-1], self.y_pred[-1])
            accuracy_task.append(accuracy_sample)
            self.all_accuracies.append(accuracy_sample)
            print(f"Accuracy score for sample {sample_inx+1}:", round(accuracy_sample, 2))
            in_accuracy_sample = self.in_accuracy_score(self.y_true[-1], self.y_pred[-1])
            in_accuracy_task.append(in_accuracy_sample)
            self.all_in_accuracies.append(accuracy_sample)
            print(f"The model's output containing the correct one:", round(in_accuracy_sample, 2))
            print("______________________________")
        print("\n\nModel's predictions for the sample:", "\tGOLDEN\t\tPREDICTED\n", sep="\n\n")
        y_pred_joined = [[prediction.replace("\n", " ") for prediction in part_predictions]
                         for part_predictions in self.y_pred]
        [print(f"\t{golden}\t\t{prediction}") for golden, prediction in zip(self.y_true[-1], y_pred_joined[-1])]
        accuracy = round(sum(accuracy_task) / len(accuracy_task), 2)
        print(f"Accuracy score per task {task_id}:", accuracy)
        # print("General accuracy:", round(sum(self.all_accuracies) / len(self.all_accuracies), 2))
        in_accuracy = round(sum(in_accuracy_task) / len(in_accuracy_task), 2)
        print(f"The model's output containing the correct one:", in_accuracy)
        # print("Generally, the model's output contains the correct one:",
        # round(sum(self.all_in_accuracies) / len(self.all_in_accuracies), 2))

        print(f"The work on task {task_num} is finished successfully")

        # [result.extend([true, pred]) for true, pred, result in zip(self.y_true[-1], self.y_pred[-1], task_results)]
        # TODO into config: task name
        # TODO: add checking the true answer "in" in the predicted one
        with open("results/prompt_11_.csv", "a+", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            task_results[0].extend([accuracy, in_accuracy])
            writer.writerows(task_results)

        print(f"The results of task {task_num} are saved successfully")


if __name__ == "__main__":
    # TODO: add more description to the outputs
    # TODO: add to config if we want to print certain data (level of messages: INFO, CRITICAL and so on)
    qa_task_iterate = QATaskIterate()
    # TODO: separate loading the model
    print("The model is loaded successfully")
    # into config: run level
    train_data, test_data = (
        qa_task_iterate.make_train_data(),
        qa_task_iterate.make_test_data(),
    )
    print("The data is loaded successfully", end="\n\n")
    assert len(train_data.items()) == len(test_data.items())
    # into config: output path
    headers = ["task_id", "sample_no", "task", "true_result", "model_result", "accuracy", "in_accuracy"]
    with open("results/prompt_11_.csv", "a+", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")

    print("Starting to query the model", end="\n\n")
    # TODO into config: task ids
    for task_num in range(1, 21):
        qa_task_iterate.make_task_data(test_data, task_num)  # Task n°1
        qa_task_iterate.iterate_context(task_num)
        print("______________________________", end="\n\n")

    with open("results/prompt_11_.csv", "a+", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        row = [""] * (len(headers) - 2)
        general_accuracy = round(sum(qa_task_iterate.all_accuracies) / len(qa_task_iterate.all_accuracies), 2)
        print("General accuracy:", general_accuracy)
        general_in_accuracy = round(sum(qa_task_iterate.all_in_accuracies) / len(qa_task_iterate.all_in_accuracies), 2)
        print("Generally, the model's output contains the correct one:", general_in_accuracy)
        writer.writerow(row + [general_accuracy, general_in_accuracy])
