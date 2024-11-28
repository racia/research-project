import os
import pandas as pd
import re
from typing import List, Dict
import torch
# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score


class QATaskIterate():
    def __init__(self):

        self.data_dir = "tasks_1-20_v1-2/en/"

        self.token = os.getenv("HUGGINGFACE")

        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Use the appropriate model name from the Hugging Face Hub
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        #pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")

        self.prompt = [{"role": "system", "content": """You will be given a sequence of context sentences 
        with questions included. Please answer each such question by only providing the 
        correct answer. For example: The correct answer to: John is on the playground. 
        Mary is in the kitchen. Where is John? is: Playground"""},]
        
        self.y_true = []
        self.y_pred = []

    def get_prompt(self):
        return self.prompt

    def make_test_data(self):
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
    
    def make_train_data(self):
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

    def make_task_data(self, data, task_id):
        task_data = []
        # Returns a list of dict of one sequence of context sentences
        with open(self.data_dir+data[task_id]["test"], "r") as f:
            d = 0
            task_data.append({})
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                iter, cont = re.split(" ", line, maxsplit=1) # Remove newline
                if len(cont.split("\t"))>2:
                    task_data[d][int(iter)] = cont.split("\t")
                else:
                    task_data[d][int(iter)] = cont
                if "?" in line and i!=(len(lines)-1):
                    task_data.append({})
                    d+=1
        return task_data

    def make_task_example(self, task_data, exp_id):
        # Return a dict with "role" - "content" with one entry of context sentence sequence 
        # to append to current pipeline
        sample = list(task_data[exp_id].values())
        context = sample[:-1] # Strings of context
        answer = sample[-1] # List with question and final answer
        x = " ".join(context)
        y = answer[0] # Take only question
        self.y_true.append(answer[1])
        return {"role": "user", "content": x+" "+y}

    def iterate_context(self, prompt, task_data):
        for i in range(5):
            # 1. Add sample
            prompt.append(self.make_task_example(task_data, i))
            # 2. Create generation prompt
            formatted_prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            print("Formatted prompt:\n", formatted_prompt)
            # 3. Tokenize
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)
            inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
            print("Tokenized inputs:\n", inputs)
            # 4. Generate text
            outputs = self.model.generate(**inputs, max_new_tokens=12, temperature=0.1, pad_token_id=self.tokenizer.eos_token_id)
            #print(outputs)
            # 5. Decode output back to string
            decoded_output = self.tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
            #ans = response[0]["generated_text"][-1]["content"] # Obtain answer
            self.y_pred.append(decoded_output.lower())
            print(prompt[-1]["content"], decoded_output) # Print qa
            #prompt = decoded_output #Update history
        print(self.y_true, self.y_pred)
        print(accuracy_score(self.y_true, self.y_pred))


if __name__ == "__main__":
    qa_task_iterate = QATaskIterate()
    prompt = qa_task_iterate.get_prompt()
    train_data, test_data = qa_task_iterate.make_train_data(), qa_task_iterate.make_test_data()
    assert len(train_data.items()) == len(test_data.items())
    #print(test)
    for i in range(1, 3):    
        task_data = qa_task_iterate.make_task_data(test_data, i) # Task n°1
        qa_task_iterate.iterate_context(prompt, task_data)
