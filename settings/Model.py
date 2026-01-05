from __future__ import annotations

import os
import warnings

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
        k: int | None = None,
        p: float | None = None,
        mode: Mode = "eval",
        wrapper: Wrapper = None,
        interpretability: bool = None,
    ):
        self.token: str = os.getenv("HUGGINGFACE")
        self.name: str = name
        self.role = role
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.to_continue: bool = to_continue
        self.mode: Mode = mode

        self.model, self.tokenizer = self.load()

        if interpretability:
            self.interpretability = Interpretability()
            self.interpretability.tokenizer = self.tokenizer

        self.wrapper = encode_wrapper(wrapper, self.tokenizer) if wrapper else None
        self.chat: Chat = None
        self.k = k
        self.p = p

    def load(self) -> tuple[OpenLlamaPreTrainedModel, PreTrainedTokenizerFast]:
        """
        Load the model and the tokenizer.
        Set the model in mode.
        The model is loaded with memory optimizations.

        :return: tuple: model, tokenizer
        """
        print(
            f"The model {self.name} is being loaded in mode '{self.mode}'...",
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
            "attn_implementation": "eager",
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

        print(f"The model {self.name} was loaded successfully", flush=True)

        return model, tokenizer

    def call(
        self,
        data: SamplePart | str = None,
        from_chat: bool = False,
        to_continue: bool = False,
        filter_eot: bool = True,
    ) -> tuple[str, InterpretabilityResult]:
        """
        Calls the model with memory optimizations and optionally with Interpretability (depends on config).
        The user message is added to the chat if a part or a formatted prompt is provided.
        Otherwise, the model is called on the current chat as is. The generated model message is added always.
        Whenever a string message is added, it is encoded, so the chat assumes all the encoded ids to be in place.

        :param data: The data to be used for the model call. It can be a SamplePart or a string.
        :param from_chat: Whether the message is from the chat or not
        :param to_continue: Whether the model should continue the last message or not
        :param filter_eot: Whether to filter the <|eot_id|> token from the end of the output or not
        :return: The decoded model output
        """
        if not self.chat:
            raise ValueError(
                "Chat is not set. Please set the chat before calling the model."
            )
        if data and not from_chat:
            self.chat.add_message(
                part=data,
                source=Source.user,
                wrapper=self.wrapper,
            )
        elif not (data or from_chat):
            raise ValueError(
                "Either data or from_chat should be set. Please set one of them."
            )

        with torch.no_grad():
            call_from_part = type(data) is SamplePart and not from_chat
            # includes flat ids for all the messages in the chat, including the wrapper
            chat_ids = self.chat.convert_into_datatype(
                datatype="ids",
                identify_target=True if call_from_part else False,
                to_continue=to_continue,
            )

            inputs = {"input_ids": chat_ids.to("cuda")}
            torch.cuda.empty_cache()

            with autocast("cuda"):
                # includes all the previous ids + the model output
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True if self.temperature > 0 else False,
                    use_cache=False,
                    num_beams=1,  # no beam search, reduce GPU memory usage
                )
                encoded_output = outputs[0][inputs["input_ids"].size(1) :]

                # remove eot token if it is at the end of the output
                if filter_eot:
                    eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    # remove trailing spaces at the end of the output
                    while (
                        len(encoded_output) > 0
                        and self.tokenizer.decode([encoded_output[-1]]).isspace()
                    ):
                        encoded_output = encoded_output[:-1]

                    # remove eot token if it is at the end of the output
                    if len(encoded_output) > 0 and encoded_output[-1] == eot:
                        encoded_output = encoded_output[:-1]

                    if len(encoded_output) == 0:
                        warnings.warn(
                            "DEBUG: The model output is empty after filtering the <|eot_id|> token. Using empty string as output."
                        )
                        # TODO: encoded_output = [] instead?
                        encoded_output = self.tokenizer.convert_tokens_to_ids("''")

                decoded_output = self.tokenizer.decode(encoded_output).strip()

                print(f"DEBUG: Model output (decoded): {decoded_output}", flush=True)

                # the model expanded on the message, so we need to update it
                if to_continue:
                    self.chat.adjust_message(
                        decoded_output, encoded_output, full_output=False
                    )
                else:
                    self.chat.add_message(
                        part=decoded_output, source=Source.assistant, ids=encoded_output
                    )

                interpretability_result = None

                if self.role == "student" and not self.interpretability:
                    raise ValueError("Interpretability is not set for student model!")

                if self.role != "teacher" and self.interpretability and decoded_output:
                    try:
                        # output tensor includes all the previous ids + the model output
                        output_tensor = self.model(
                            outputs,
                            return_dict=True,
                            output_attentions=True,
                            output_hidden_states=False,
                        )
                        print("Output tensor attentions:", output_tensor["attentions"])
                        if type(data) is not SamplePart:
                            raise TypeError(
                                "For interpretability plotting, data should be of type SamplePart"
                            )
                        # for the settings, the final model output is currently not plotted
                        interpretability_result = (
                            self.interpretability.process_attention(
                                output_tensor=output_tensor,
                                chat=self.chat,
                                chat_ids=outputs,
                            )
                        )
                    except torch.OutOfMemoryError:
                        warnings.warn(
                            "DEBUG: Out of memory error while calculating interpretability scores * before *. "
                            "Skipping this step."
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
            with autocast("cuda"):
                teacher_outputs = self.model(input_ids)
                teacher_logits = teacher_outputs.logits
                teacher_probs = torch.nn.functional.softmax(
                    teacher_logits[:, -1, :], dim=-1
                )
            torch.cuda.empty_cache()
        return teacher_probs
