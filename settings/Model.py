import os

import torch
from torch.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from inference.DataLevels import SamplePart
from settings.config import Mode


class Model:
    """
    Class for the model with memory optimizations.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        to_continue: bool,
        mode: Mode = "eval",
    ):
        self.token: str = os.getenv("HUGGINGFACE")
        self.model_name: str = model_name
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.to_continue: bool = to_continue
        self.mode: Mode = mode
        self.model, self.tokenizer = self.load()

        self.curr_sample_part: SamplePart = None

    def load(self) -> tuple:
        """
        Load the model and the tokenizer.
        Set the model in mode.
        The model is loaded with memory optimizations.

        :return: tuple: model, tokenizer
        """
        print(
            f"The model {self.model_name} is being loaded in mode '{self.mode}'",
            end="\n\n",
            flush=True,
        )

        # create an offload folder
        if not os.path.exists("offload_folder"):
            os.makedirs("offload_folder")

        # quantisation config for memory optimizations
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # set model kwargs for more memory optimizations
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,
            "offload_folder": "offload_folder",
            "offload_state_dict": True,
        }

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        if self.mode == "eval":
            model.eval()
        elif self.mode == "train":
            model.train()
        else:
            raise ValueError(f"The mode '{self.mode}' is doesn't exist.")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id

        torch.cuda.empty_cache()

        print(f"The model {self.model_name} was  loaded successfully", flush=True)

        return model, tokenizer

    def call(self, prompt: str) -> str:
        """
        Calls the model with memory optimizations.
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=2048,
                add_special_tokens=True,
            )
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            torch.cuda.empty_cache()

            with autocast("cuda"):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True if self.temperature > 0 else False,
                    use_cache=True,
                    num_beams=1,  # no beam search, reduce GPU memory usage
                )
                input_length = inputs["input_ids"].size(1)
                decoded_output = self.tokenizer.decode(
                    outputs[0][input_length:],
                    skip_special_tokens=True,
                ).lower()

            torch.cuda.empty_cache()
        return decoded_output

    def call_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Call the model and get its logit output.
        Using these logits, the probabilities are calculated.

        :param input_ids: the input ids of the model

        :return: the probabilities of the model
        """
        with torch.no_grad():
            teacher_outputs = self.model(input_ids)
            teacher_logits = teacher_outputs.logits
            teacher_probs = torch.nn.functional.softmax(
                teacher_logits[:, -1, :], dim=-1
            )
        return teacher_probs
