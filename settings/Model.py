from __future__ import annotations

import os

import torch
from torch.amp import autocast
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    OpenLlamaPreTrainedModel,
    PreTrainedTokenizerFast,
)

from inference.Chat import Chat, Source
from inference.DataLevels import SamplePart
from interpretability.Interpretability import Interpretability
from interpretability.utils import InterpretabilityResult
from settings.config import Mode, Wrapper
from settings.utils import encode_wrapper


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
        role: str | None,
        wrapper: Wrapper = None,
        mode: Mode = "eval",
        interpretability: Interpretability = None,
    ):
        self.token: str = os.getenv("HUGGINGFACE")
        self.name: str = name
        self.role = role
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.to_continue: bool = to_continue
        self.mode: Mode = mode
        self.model, self.tokenizer = self.load()
        self.interpretability = interpretability
        self.interpretability.tokenizer = self.tokenizer

        self.wrapper = encode_wrapper(wrapper, self.tokenizer)
        self.chat: Chat = None

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
        self,
        part: SamplePart = None,
        formatted_prompt: str = None,
    ) -> tuple[str, InterpretabilityResult]:
        """
        Calls the model with memory optimizations and optionally with Interpretability.
        :param part: The current sample part
        :param formatted_prompt: The formatted prompt including the whole chat
        :return: The decoded model output
        """
        if not self.chat:
            raise ValueError(
                "Chat is not set. Please set the chat before calling the model."
            )
        if part and formatted_prompt:
            raise ValueError(
                "Either part or formatted_prompt should be provided, not both."
            )
        if formatted_prompt and self.interpretability:
            raise ValueError(
                "Interpretability cannot be calculated with formatted_prompt."
            )
        print("Wrapper before model call")
        print(self.wrapper)
        self.chat.add_message(part=part, source=Source.user, wrapper=self.wrapper)

        with torch.no_grad():
            device = next(self.model.parameters()).device

            if self.interpretability:
                # self.prepare_prompt(chat=chat)
                chat_ids = self.chat.chat_to_ids()
                print(
                    f"Formatted prompt (to remove):",
                    self.tokenizer.batch_decode(chat_ids),
                    sep="\n",
                    end="\n",
                )
                print(chat_ids)
                inputs = {"input_ids": chat_ids.to("cuda")}
                print("inputs", inputs)
            # if self.chat:
            #     chat_ids, context_sent_spans, sys_prompt_len = (
            #         self.chat.convert_into_ids_old(
            #             chat_part=self.chat.messages,
            #             tokenizer=self.tokenizer,
            #         )
            #     )
            #     inputs = {k: v.to(device) for k, v in chat_ids.items()}
            else:
                inputs = self.tokenizer(
                    formatted_prompt,
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

                encoded_output = outputs[0][inputs["input_ids"].size(1) :]
                decoded_output = self.tokenizer.decode(
                    encoded_output, skip_special_tokens=True
                ).lower()

                print("decoded_output", decoded_output)

                self.chat.add_message(
                    part=decoded_output, source=Source.assistant, ids=encoded_output
                )

                interpretability_result = None
                if self.interpretability and decoded_output:
                    output_tensor = self.model(
                        outputs,
                        return_dict=True,
                        output_attentions=True,
                        output_hidden_states=False,
                    )

                    interpretability_result = self.interpretability.calculate_attention(
                        part=part,
                        after=not part.multi_system,
                        chat_ids=outputs,
                        context_sent_spans=self.chat.sent_spans,
                        output_tensor=output_tensor,
                        model_output_len=len(encoded_output),
                        sys_prompt_len=len(self.chat.system_message["ids"]),
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
