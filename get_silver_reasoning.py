import gc
import os.path
import re
import warnings
from pathlib import Path

import pandas as pd
from torch import autocast

from data.DataLoader import DataLoader
from inference.Chat import Chat
from inference.Prompt import Prompt
from inference.utils import Source
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
    overwrite = True
    gc.collect()

    # Load the data
    data_loader = DataLoader()
    # without any flags should receive data structured in levels
    data = data_loader.load_task_data(
        path=f"{home}/tasks_1-20_v1-2/en-valid/",
        split=split,
        tasks=[3, 17, 20],
        multi_system=False,
    )

    model = Model(
        "meta-llama/Meta-Llama-3-70B-Instruct",
        max_new_tokens=300,
        temperature=0.1,
        to_continue=False,
        role="model",
    )

    # Load the prompt
    prompt = Prompt(
        prompt_path=f"{home}/research-project/inference/prompts/skyline_reasoning.txt",
        tokenizer=model.tokenizer,
    )

    # Process only tasks assigned to this GPU
    for task_id, task in sorted(data.items()):
        output_file = f"{home}/research-project/data/silver_reasoning/silver_reasoning_{split}_{task_id}.csv"

        print(f"Writing to {output_file}")
        print(f"Processing Task {task_id}...", flush=True)

        if not os.path.exists(output_file) or overwrite is True:
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

    eot = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    for sample_part_idx, sample_part in enumerate(sample_parts):
        print(f"Sample Part: {sample_part}")

        # Format the prompt
        formatted_prompt_str = prompt.text.format(
            context=sample_part.structured_context,
            question=sample_part.structured_question,
            answer=sample_part.golden_answer,
        )

        print(f"Formatted Prompt: {formatted_prompt_str}")

        model.chat.add_message(
            part=formatted_prompt_str,
            source=Source.user,
        )

        chat_ids = model.chat.convert_into_datatype(
            datatype="ids",
            identify_target=False,
            to_continue=False,
        )

        inputs = {"input_ids": chat_ids.to("cuda")}
        torch.cuda.empty_cache()

        with torch.no_grad(), autocast("cuda"):
            outputs = model.model.generate(
                **inputs,
                max_new_tokens=model.max_new_tokens,
                temperature=model.temperature,
                pad_token_id=model.tokenizer.eos_token_id,
                do_sample=True if model.temperature > 0 else False,
                use_cache=True,
                num_beams=1,  # no beam search, reduce GPU memory usage
            )

            encoded_output = outputs[0][inputs["input_ids"].size(1) :]

            # remove eot token if it is at the end of the output
            if len(encoded_output) > 0 and encoded_output[-1] == eot:
                encoded_output = encoded_output[:-1]

            if len(encoded_output) == 0:
                warnings.warn(
                    "DEBUG: The model output is empty after filtering the <|eot_id|> token. Using empty string as output."
                )
                encoded_output = []

        decoded_output = model.tokenizer.decode(encoded_output).strip()

        print(f"Decoded Output: {decoded_output}")

        reasoning_pattern = re.compile(r"(?i)reasoning:[\s ]*(.+)")
        reasoning_search = reasoning_pattern.search(decoded_output)
        reasoning = reasoning_search[1].strip() if reasoning_search else decoded_output

        print(
            f"Task {task_id}, Sample {sample_id}, Part {sample_part_idx}: Reasoning extracted",
            flush=True,
        )

        new_row = pd.DataFrame(
            [
                {  # all counters should start at 1 -> addition one to each
                    "id_": id_counter + 1,
                    "task_id": task_id,
                    "sample_id": sample_id + 1,
                    "part_id": sample_part_idx + 1,
                    "context": sample_part.structured_context,
                    "question": sample_part.structured_question,
                    "golden_answer": sample_part.golden_answer,
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
