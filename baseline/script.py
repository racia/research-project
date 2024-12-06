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
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Use the appropriate model name from the Hugging Face Hub
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")

        # TODO into config: prompt
        self.prompt = [{"role": "system", "content": """You will be given a sequence of context sentences and \
then asked questions about them. Please answer each question concisely and to your best abilities. For example:
Context sentences: John is on the playground. Mary is in the kitchen.
Question: Where is John?
Answer: Playground"""},]

        self.y_true, self.y_pred = [], []
        self.task_data = []

    def get_prompt(self) -> List[Dict[str, str]]:
        return self.prompt

    # TODO: generalize the three functions into one (make_data with possible parameters: train, eval, test)
    def make_test_data(self) -> Dict[int, Dict[str, List[str]]]:
        test_data = {}
        for r, d, f in os.walk(self.data_dir):
            #        print(r,d,f)
            for i in range(1, 21):
                pattern_test = re.compile(f"qa{i}_[\w-]+_test\.txt")
                test = list(filter(lambda x: re.findall(pattern_test, x), f))
                print(f"Loaded Test Data: {test[0]}\n")
                test_data[i] = {}
                test_data[i]["test"] = test[0]
        return test_data

    def make_eval_data(self):
        pass

    def make_train_data(self) -> Dict[int, Dict[str, List[str]]]:
        train_data = {}
        for r, d, f in os.walk(self.data_dir):
            #        print(r,d,f)
            for i in range(1, 21):
                pattern_train = re.compile(f"qa{i}_[\w-]+_train\.txt")
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
        self.task_data = []
        with open(self.data_dir + data[task_id]["test"], "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.task_data.append({})
            for i, line in enumerate(lines):
                line = line.strip()  # Remove newlines
                sentence_id, sentence = line.split(maxsplit=1)
                sentence_is_question = "\t" in sentence
                if sentence_is_question:
                    question_line = sentence.split("\t")
                    self.task_data[-1][int(sentence_id)] = question_line.pop(0)  # add the question to the sample
                    # sentence ids start from 1, so we can reserve 0 for question answers
                    self.task_data[-1][0] = question_line  # add the answer and the references with a separate key 0
                else:
                    self.task_data[-1][int(sentence_id)] = sentence
                if sentence_is_question and i != (len(lines) - 1):
                    self.task_data.append({})
        return self.task_data

    def make_task_example(self, exp_id: int, to_enumerate:bool=True) -> Dict[str, str]:
        """

        Unused value - references

        :param exp_id:
        :param to_enumerate:
        :return: a dict with "role" - "content" with one entry of context sentence sequence
        """
        sample = self.task_data[exp_id]  # context sentences and the question in the end
        references = sample.pop(0)  # Removes the line with the answer and the references from the sample dict
        answer = references.pop(0)  # Separates the answer from the references, the references now contains only refs

        if to_enumerate:
            x = "\n".join([". ".join(i_sentence) for i_sentence in list(sample.items())])
        else:
            sample = list(sample.values())  # Strings of context and the question
            x = "\n".join(sample)

        print("Sample:\n", sample, "\n")
        self.y_true.append(answer)  # Add the answer to y_true (without the references)
        return {"role": "user", "content": x}

    def iterate_context(self, prompt, task_num, sample_id):
        """

        :param prompt:
        :param task_num:
        :param sample_id:
        :return:
        """
        task_results = []
        print("Starting to query the model")
        # TODO into config: sample
        samples_per_task = 5
        for i in range(samples_per_task):
            sample_id += 1
            sample_result = []
            # TODO into config: number of tasks
            total_tasks = samples_per_task * 20
            print(f"\n\n-* TASK {task_num} | SAMPLE {i} | ID {sample_id}/{total_tasks} *-\n\n")
            # 1. Add sample
            task_example = self.make_task_example(i)
            prompt.append(task_example)
            sample_result.append(task_example["content"])
            task_results.append(sample_result)
            # 2. Create generation prompt
            formatted_prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            print("Formatted prompt:\n", formatted_prompt, end="\n")
            # 3. Tokenize
            inputs = self.tokenizer(
                formatted_prompt, return_tensors="pt", add_special_tokens=True
            )
            inputs = {
                key: tensor.to(self.model.device) for key, tensor in inputs.items()
            }
            display_inputs = {k: v[0].tolist() for k, v in inputs.items()}
            print("Tokenized inputs:\n", json.dumps(display_inputs, indent=4), end="\n\n")
            # 4. Generate text
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=12,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # print(outputs)
            # 5. Decode output back to string
            decoded_output = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True
            )
            # ans = response[0]["generated_text"][-1]["content"] # Obtain answer
            self.y_pred.append(decoded_output.lower())
            print("Model's output:", decoded_output, end="\n\n")  # Print qa
            # prompt = decoded_output #Update history
            print("______________________________")

        print(self.y_true, self.y_pred)
        accuracy = accuracy_score(self.y_true, self.y_pred)
        print("Accuracy score:", accuracy)

        [result.extend([true, pred]) for true, pred, result in zip(self.y_true, self.y_pred, task_results)]
        # TODO into config: task name
        with open("prompt_2.csv", "a+", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            task_results[0].append(accuracy)
            writer.writerows(task_results)

        return sample_id


if __name__ == "__main__":
    # TODO: add more description to the outputs
    # TODO: add to config if we want to print certain data (level of messages: INFO, CRITICAL and so on)

    qa_task_iterate = QATaskIterate()
    print("The model is loaded successfully")
    prompt = qa_task_iterate.get_prompt()
    train_data, test_data = (
        qa_task_iterate.make_train_data(),
        qa_task_iterate.make_test_data(),
    )
    print("The data is loaded successfully")
    assert len(train_data.items()) == len(test_data.items())

    with open("prompt_2.csv", "a+", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["task", "true_result", "model_result", "accuracy"])

    sample_id = 0

    # TODO into config: task ids
    for task_num in range(1, 21):
        qa_task_iterate.make_task_data(test_data, task_num)  # Task n°1
        sample_id = qa_task_iterate.iterate_context(prompt, task_num, sample_id)
        print(f"The work on task {task_num} is finished successfully")

    print("The run is finished successfully")
