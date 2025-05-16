import gc
import os.path
import re
from pathlib import Path

import pandas as pd

from data.DataLoader import DataLoader
from inference.Chat import Chat
from inference.Prompt import Prompt
from inference.utils import numerate_lines
from settings.Model import Model

home = Path.home()

import os
import torch


def main():
    """
    Run the Skyline model with memory optimizations
    """
    split = "test"
    max_samples = 100
    torch.cuda.empty_cache()
    gc.collect()

    # Load the data
    data_loader = DataLoader()
    # without any flags should receive data structured in levels
    valid_data = data_loader.load_task_data(
        path=f"{home}/tasks_1-20_v1-2/en-valid/",
        split=split,
        tasks=[15],
        multi_system=False,
    )

    model = Model(
        "meta-llama/Meta-Llama-3-70B-Instruct",
        max_new_tokens=100,
        temperature=0.1,
        to_continue=False,
        role="model",
    )

    # Load the prompt
    prompt = Prompt(
        prompt_path=f"{home}/research-project/inference/prompts/skyline_reasoning.txt"
    )

    # Process only tasks assigned to this GPU
    for task_id, task in sorted(valid_data.items()):
        output_file = f"{home}/research-project/silver_reasoning_{split}_{task_id}.csv"

        if not os.path.exists(output_file):
            result_df = pd.DataFrame(
                columns=[
                    "id_",
                    "task_id",
                    "sample_id",
                    "part_id",
                    "context",
                    "question",
                    "golden_answer",
                    "silver_reasoning",
                ]
            )
            result_df.to_csv(output_file, index=False)

        id_counter = 0
        safety_length = min(len(task), max_samples)
        for sample_id, sample_parts in list(task.items())[:safety_length]:
            id_counter = process_sample(
                task_id, sample_id, sample_parts, prompt, model, output_file, id_counter
            )

            torch.cuda.empty_cache()
            gc.collect()


def process_sample(
    task_id, sample_id, sample_parts, prompt, model, output_file, id_counter
) -> int:
    """
    Process a single sample.

    :param task_id: the task ID
    :param sample_id: the sample ID
    :param sample_parts: the sample parts
    :param prompt: the prompt
    :param model: the model
    :param output_file: the output file
    :param id_counter: the ID counter
    """
    model.chat = Chat(
        system_prompt=prompt, model_role=model.role, tokenizer=model.tokenizer
    )

    for sample_part_idx, sample_part in enumerate(sample_parts):
        # Format prompt components
        context = numerate_lines(sample_part["context"])
        formatted_context = "\n".join(context)

        questions = list(sample_part["question"].values())
        formatted_questions = "\n".join(questions)

        answers = [" ".join(ans) for ans in sample_part["answer"].values()]
        formatted_answers = "\n".join(answers)

        # Format the prompt
        formatted_prompt_str = prompt.text.format(
            context=formatted_context,
            question=formatted_questions,
            answer=formatted_answers,
        )
        decoded_output = model.call(formatted_prompt=formatted_prompt_str)

        reasoning_pattern = re.compile(r"(?i)reasoning:[\s ]*(.+)")
        reasoning_search = reasoning_pattern.search(decoded_output)
        reasoning = reasoning_search[1].strip() if reasoning_search else decoded_output

        print(
            f"Task {task_id}, Sample {sample_id}, Part {sample_part_idx}: Reasoning extracted"
        )

        new_row = pd.DataFrame(
            [
                {  # all counters should start at 1 -> addition one to each
                    "id_": id_counter + 1,
                    "task_id": task_id,
                    "sample_id": sample_id + 1,
                    "part_id": sample_part_idx + 1,
                    "context": formatted_context,
                    "question": formatted_questions,
                    "golden_answer": formatted_answers,
                    "silver_reasoning": reasoning,
                }
            ]
        )

        write_header = (
            not os.path.exists(output_file) or os.stat(output_file).st_size == 0
        )

        new_row.to_csv(output_file, index=False, mode="a", header=write_header)

        id_counter += 1

    return id_counter


if __name__ == "__main__":
    main()
