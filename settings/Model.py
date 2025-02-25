import os

import torch
from torch.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    """
    Class for the model.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        to_continue: bool,
    ):
        self.token = os.getenv("HUGGINGFACE")
        self.model_name = model_name
        self.model, self.tokenizer = self.load()

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.to_continue = to_continue

    def load(self) -> tuple:
        """
        Load the model and the tokenizer for the instance model name.

        :return the model and the tokenizer
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        torch.cuda.empty_cache()

        return model, tokenizer

    def call(self, prompt: str) -> str:
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
            inputs = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            )
            inputs = {
                key: tensor.to(self.model.device) for key, tensor in inputs.items()
            }

            # autocast enables mixed precision for computational efficiency
            with autocast("cuda"):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

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
