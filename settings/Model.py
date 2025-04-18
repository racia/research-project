import os

import numpy as np
import torch
from torch.amp import autocast
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    OpenLlamaPreTrainedModel,
    PreTrainedTokenizerFast,
)

from inference.Chat import Chat
from inference.DataLevels import SamplePart
from interpretability.Interpretability import Interpretability
from interpretability.utils import InterpretabilityResult
from settings.config import Mode


class Model:
    """
    Class for the model with memory optimizations.
    """

    def __init__(
        self,
        name: str,
        max_new_tokens: int,
        temperature: float,
        to_continue: bool,
        mode: Mode = "eval",
        interpretability: Interpretability = None,
    ):
        self.token: str = os.getenv("HUGGINGFACE")
        self.name: str = name
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.to_continue: bool = to_continue
        self.mode: Mode = mode
        self.model, self.tokenizer = self.load()
        self.interpretability = interpretability

    def load(self) -> tuple[OpenLlamaPreTrainedModel, PreTrainedTokenizerFast]:
        """
        Load the model and the tokenizer.
        Set the model in mode.
        The model is loaded with memory optimizations.

        :return: tuple: model, tokenizer
        """
        print(
            f"The model {self.name} is being loaded in mode '{self.mode}'",
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

        model = AutoModelForCausalLM.from_pretrained(self.name, **model_kwargs)

        if self.mode == "eval":
            model.eval()
        elif self.mode == "train":
            model.train()
        else:
            raise ValueError(f"The mode '{self.mode}' is doesn't exist.")

        tokenizer = AutoTokenizer.from_pretrained(self.name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id

        torch.cuda.empty_cache()

        print(f"The model {self.name} was  loaded successfully", flush=True)

        return model, tokenizer

    def call(
        self, part: SamplePart, prompt: str, chat: Chat = None
    ) -> tuple[str, InterpretabilityResult]:
        """
        Calls the model with memory optimizations and optionally with Interpretability.
        :param part: The current sample part
        :param prompt: The formatted prompt
        :param chat: The current chat
        :return: The decoded model output
        """
        with torch.no_grad():
            device = next(self.model.parameters()).device

            if chat:
                chat_ids, context_sent_spans, sys_prompt_len = chat.convert_into_ids(
                    chat_part=chat.messages,
                    tokenizer=self.tokenizer,
                )
                inputs = {k: v.to(device) for k, v in chat_ids.items()}
            else:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=2048,
                    add_special_tokens=True,
                )
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
                encoded_output = outputs[0][input_length:]

                decoded_output = self.tokenizer.decode(
                    encoded_output, skip_special_tokens=True
                ).lower()

                interpretability_result = InterpretabilityResult(
                    np.array([]), [], [], 0
                )

                # Controlled call (i.e. generate_student() in Feedback shouldn't call ?)
                if chat and decoded_output:
                    output_tensor = self.model(
                        outputs,
                        return_dict=True,
                        output_attentions=True,
                        output_hidden_states=False,
                    )

                    interpretability_result = self.interpretability.calculate_attention(
                        tokenizer=self.tokenizer,
                        part=part,
                        after=not part.multi_system,
                        chat_ids=outputs,
                        context_sent_spans=context_sent_spans,
                        output_tensor=output_tensor,
                        model_output_len=len(encoded_output),
                        sys_prompt_len=sys_prompt_len,
                    )

            torch.cuda.empty_cache()

        return decoded_output, interpretability_result

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
