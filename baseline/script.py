import os
import pandas as pd
import re
from typing import List, Dict
import torch
# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer

token = os.getenv("HUGGINGFACE")

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Use the appropriate model name from the Hugging Face Hub
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")


prompt = [{"role": "system", "content": """You will be given a sequence of context sentences 
with questions included. Please answer each such question by only providing the 
correct answer. For example: The correct answer to: John is on the playground. 
Mary is in the kitchen. Where is John? is: Playground"""},]


def make_data():
    train_data, test_data = {}, {}
    for r, d, f in os.walk("tasks_1-20_v1-2/en/"):
#        print(r,d,f)
        for i in range(1, 21):
            pattern_train = re.compile(f"qa{i}_[\w-]+_train\.txt")
            pattern_test = re.compile(f"qa{i}_[\w-]+_test\.txt")
            train = list(filter(lambda x: re.findall(pattern_train, x), f))
            test = list(filter(lambda x: re.findall(pattern_test, x), f))
            print(f"Loaded Train Data: {train[0]}\n", f"Test Data: {test[0]}\n")
            train_data[i], test_data[i] = {}, {}
            train_data[i]["train"] = train[0]
            test_data[i]["test"] = test[0]
    assert len(train_data.items()) == len(test_data.items())
    return train_data, test_data


def make_task_data(data, task_id):
    task_data = []
    # Returns a list of dict of one sequence of context sentences
    with open(f"tasks_1-20_v1-2/en/"+data[task_id]["test"], "r") as f:
        d = 0
        task_data.append({})
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            iter, cont = re.split(" ", line, maxsplit=1) # Remove newline
#            print(iter, cont)
            if len(cont.split("\t"))>2:
                task_data[d][int(iter)] = cont.split("\t")
            else:
                task_data[d][int(iter)] = cont
            if "?" in line and i!=(len(lines)-1):
                task_data.append({})
                d+=1
    return task_data

def make_task_example(task_data, exp_id):
    # Return a dict with "role" - "content" with one entry of context sentence sequence 
    # to append to current pipeline
    sample = list(task_data[exp_id].values())
    context = sample[:2] # Strings of context
    answer = sample[2] # List with question and final answer
    x, y = " ".join(context + list(answer[0])) # Take only question
    return {"role": "user", "content": x+y}

def iterate_context(prompt, task_data):
    for i in range(5):
        # 1. Add sample
        prompt.append(make_task_example(task_data, i))
        # 2. Create generation prompt
        formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        print("Formatted prompt:\n", formatted_prompt)
        # 3. Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        print("Tokenized inputs:\n", inputs)
        # 4. Generate text
        outputs = model.generate(**inputs, max_new_tokens=12, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
        #print(outputs)
        # 5. Decode output back to string
        decoded_output = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        #ans = response[0]["generated_text"][-1]["content"] # Obtain answer
        print(prompt[-1]["content"], decoded_output) # Print qa
        #prompt = decoded_output #Update history

train, test = make_data()
#print(test)
task_data = make_task_data(test, 1) # Task n°1
iterate_context(prompt, task_data)
