from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Union

import torch
from transformers import PreTrainedTokenizerFast

from inference.DataLevels import SamplePart
from inference.Prompt import Prompt
from inference.utils import generation_tokens, sents_to_ids, upd_span


@dataclass
class Source:
    """
    This class handles the roles of the participants in the conversation.
    """

    system: str = "system"
    user: str = "user"
    assistant: str = "assistant"

    options = (system, user, assistant)


class Chat:
    """
    This class handles the chats with the model.
    """

    def __init__(
        self,
        model_role: str,
        system_prompt: Prompt = None,
        tokenizer: PreTrainedTokenizerFast = None,
    ):
        """
        Create a chat.
        A chat consists of the prompts the model is prompted with and the answers of the model.
        The prompts and answers are saved in a list.

        :param system_prompt: the first prompt the model is prompted with
        :param tokenizer: the tokenizer to encode the messages
        """
        self.model_role = model_role
        self.tokenizer = tokenizer
        self.offset = len(system_prompt.ids)

        self.system_message = {
            "role": Source.system,
            "content": system_prompt.text,
            "original_content": system_prompt.original_text,
            "ids": system_prompt.ids,
            "sent_spans": system_prompt.sent_spans,
            "spans_ids": {
                "sys": dict(zip(system_prompt.orig_sent_spans, system_prompt.orig_ids)),
                "ex": dict(
                    zip(
                        [
                            upd_span(span, self.offset)
                            for span in system_prompt.ex_sent_spans
                        ],
                        system_prompt.ex_ids,
                    )
                ),
            },
        }
        self.messages = [self.system_message]
        self.sent_spans = {}

    def __repr__(self) -> str:
        return "\n".join(
            [f"<CHAT {self.model_role}>"]
            + [f"{message['role']}: {message['content']}" for message in self.messages]
        )

    def add_message(
        self,
        part: SamplePart | str,
        source: Union[Source.user, Source.assistant],
        ids: torch.Tensor = None,
        wrapper: dict[str, dict] = None,
    ) -> None:
        """
        Add a message to the messages list. It can add either a message from the part task or assistant output.
        The message is converted into ids using the tokenizer, but the ids can also be provided for the assistant output.
        If the wrapper is present, it takes into account its ids and spans to offset the sentence spans and adds it
        to the task ids.

        :param part: part of a sample as a string or a list of strings
        :param source: the producer of the message
        :param ids: the ids of the message, if None, the ids are generated from the message
        :param wrapper: the wrapper ids and sentence spans of the message
        """
        if wrapper and ids is not None and source == Source.assistant:
            raise ValueError(
                "Wrapper can only be used for the messages created from scratch, and now, ids are passed."
            )

        spans_ids = {}
        target_sent_spans = {}
        if ids is None and not wrapper:
            ids, sent_spans = sents_to_ids(
                part.unwrapped_task.split("\n"), self.tokenizer
            )
            spans_ids["task"] = dict(
                zip([upd_span(span, self.offset) for span in sent_spans], ids)
            )
            for i, span in enumerate(sent_spans):
                if i in part.supporting_sent_inx:
                    target_sent_spans[upd_span(span, self.offset)] = ids
        elif wrapper:
            ids, sent_spans = [], []

            for key, wrap in wrapper.items():
                intro, outro = wrap["before"], wrap.get("after", wrap["before"])
                spans_ids["task"], spans_ids["wrap"] = {}, {}
                to_encode = None
                if key == "context":
                    to_encode = part.structured_context.split("\n")
                elif key == "question":
                    to_encode = [part.structured_question]

                print("context len", len(to_encode))
                to_insert_ids, to_insert_spans = sents_to_ids(to_encode, self.tokenizer)
                print("encoded message", *zip(to_insert_spans, to_insert_ids), sep="\n")

                for span, line_num in zip(to_insert_spans, part.context_line_nums):
                    if line_num in part.supporting_sent_inx:
                        target_sent_spans[upd_span(span, self.offset)] = to_insert_ids

                if to_insert_ids:
                    # all ids
                    for chunk in [intro["ids"], *to_insert_ids, outro["ids"]]:
                        print(chunk)
                        ids.extend(chunk)

                    # before wrapper spans/ids
                    print("spans", intro["sent_spans"])
                    spans_ids["wrap"][upd_span(intro["sent_spans"], self.offset)] = (
                        intro["ids"]
                    )
                    print("upd spans", upd_span(intro["sent_spans"], self.offset))

                    self.offset += len(intro["ids"])

                    # context/question spans/ids
                    for span, ids_ in zip(to_insert_spans, to_insert_ids):
                        print("span", span)
                        sent_spans.append(upd_span(span, self.offset))
                        spans_ids["task"][upd_span(span, self.offset)] = ids_
                        print("upd spans", upd_span(span, self.offset))

                        self.offset += len(ids_)

                    # after wrapper spans/ids
                    print("spans", outro["sent_spans"])
                    spans_ids["wrap"][upd_span(outro["sent_spans"], self.offset)] = (
                        outro["ids"]
                    )
                    print("upd spans", upd_span(outro["sent_spans"], self.offset))

                    self.offset += len(outro["ids"])

                    # TODO: add space before question (at the end of each chunk?)
                else:
                    # if the key is not context or question, we just add the before ids
                    # (no after because there's no formatting for reasoning nor answer)
                    ids.extend(intro["ids"])
                    self.offset += len(intro["ids"])
                    sent_spans.append(upd_span(intro["sent_spans"], self.offset))
                    spans_ids["wrap"][upd_span(intro["sent_spans"], self.offset)] = (
                        intro["ids"]
                    )

        else:
            # for the assistant output, the ids are passed
            sent_spans = [(0, len(ids))]
            ids = ids.tolist()
            print("ids", ids, type(ids))
            spans_ids["ans"] = {sent_spans[0]: ids}

        print(self.system_message["sent_spans"])
        print("ids", len(ids), ids)
        print("sent_spans", sent_spans)
        print("spans_ids", spans_ids)

        part_dict = {
            "role": source,
            "content": part if isinstance(part, str) else part.task,
            "original_content": part.unwrapped_task,
            "ids": ids,
            "sent_spans": sent_spans,
            "spans_ids": spans_ids,
            "target_sent_spans": target_sent_spans,
        }
        self.messages.append(part_dict)

    def get_sentence_spans(self, target: bool = False) -> list[tuple[int, int]]:
        """
        Get the sentence spans of the chat messages.
        :return: list of sentence spans
        """
        spans = []
        for message in self.messages:
            if target:
                spans.extend(message["target_sent_spans"])
            else:
                spans.extend(message["sent_spans"])
        return spans

    def chat_to_ids(self, max_length: int = 2048) -> torch.LongTensor:
        """
        Converts the chat into ids using the tokenizer.

        :param max_length: the maximum length of the input
        :return: list of ids
        """
        chat_ids = [self.tokenizer.convert_tokens_to_ids("<|begin_of_text|>")]
        # including the system prompt
        for i, message in enumerate(self.messages):
            print(message)
            print(message["ids"])
            message_ids = generation_tokens(self.tokenizer, message["role"])
            message_ids.extend(message["ids"])
            self.sent_spans[i] = message["sent_spans"]

            conversation_length = len(chat_ids) + len(message_ids)
            if conversation_length > max_length:
                warnings.warn(
                    f"Exceeded max length with {conversation_length}. Truncating the input."
                )
                break

            chat_ids.extend(message_ids)

        chat_ids.extend(generation_tokens(self.tokenizer, "assistant"))
        print("chat_ids", len(chat_ids), chat_ids)

        return torch.LongTensor([chat_ids])

    def convert_into_ids_old(
        self,
        tokenizer: PreTrainedTokenizerFast,
        chat_part: list[dict] = None,
        max_length: int = 2048,
    ) -> tuple[dict[str, torch.LongTensor], list[tuple[int, int]], int]:
        """
        Converts either all the chat messages or the specified ones into ids ensuring that the input does not exceed
        the max_length. The system prompt is always included in the input, regardless of the chat_part.
        The assistant token id is always added at the end of the input.

        (Partly taken from https://arxiv.org/abs/2402.18344)

        :param tokenizer: tokenizer to use
        :param chat_part: chat part to convert into ids, if None, all messages are used
        :param max_length: default max_length of model config
        :return: tensor of input tokens, supporting sentence spans and system prompt length
        """
        sys_prompt_len = 0
        history_ids = []
        supporting_sent_spans = []

        for i, message in enumerate(chat_part if chat_part else self.messages):
            if message["role"] in ["system", "user"]:
                # TODO: add begin of text token
                message_ids = generation_tokens(tokenizer, message["role"])
            else:
                message_ids = []

            for sentence in message["original_content"].split("\n"):
                # \n\n in source produces empty sentences
                if not sentence or sentence.isspace():
                    warnings.warn("Empty sentence detected.")
                    continue
                tokenized_sentence = tokenizer.encode(
                    sentence,
                    add_special_tokens=False,
                    return_tensors="pt",
                )[0]
                torch.cuda.empty_cache()
                tokenized_sentence = tokenized_sentence.tolist()

                if message["role"] == "user":
                    start = len(message_ids) + 1
                    message_ids.extend(tokenized_sentence)
                    end = (
                        len(message_ids) - 1 if i == 1 else len(message_ids)
                    )  # Correct index offset by 1
                    supporting_sent_spans.append((start, end))
                else:
                    message_ids.extend(tokenized_sentence)
            if message["role"] == "system":
                sys_prompt_len = len(message_ids)

            if len(history_ids) + len(message_ids) <= max_length:
                history_ids += message_ids
            else:
                break

        history_ids.append(tokenizer.convert_tokens_to_ids("assistant"))

        return (
            {"input_ids": torch.LongTensor([history_ids])},
            supporting_sent_spans,
            sys_prompt_len,
        )
