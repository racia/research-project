from __future__ import annotations

import warnings
from typing import Union

import torch
from transformers import PreTrainedTokenizerFast

from inference.DataLevels import SamplePart
from inference.Prompt import Prompt
from inference.utils import (
    flatten,
    get_generation_tokens,
    sents_to_ids,
    upd_span,
    Source,
)


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

        sys_prompt_spans_types = {span: "sys" for span in system_prompt.orig_sent_spans}
        example_spans_types = {span: "ex" for span in system_prompt.ex_sent_spans}

        self.system_message = {
            "role": Source.system,
            "content": system_prompt.text,
            "original_content": system_prompt.original_text,
            "tokens": system_prompt.tokens,
            "ids": system_prompt.ids,
            "spans_types": {**sys_prompt_spans_types, **example_spans_types},
        }
        self.offset = len(flatten(system_prompt.ids))
        self.messages = [self.system_message]

        self.supp_sent_spans = []
        self.part = None

    def __repr__(self) -> str:
        return "\n".join(
            [f"<CHAT {self.model_role}>"]
            + [f"{message['role']}: {message['content']}" for message in self.messages]
        )

    def remove_message(self, i: int) -> None:
        """
        Remove a message from the messages list.

        :param i: the index of the message to remove
        :return: None
        """
        if not self.messages:
            raise ValueError("No messages to remove.")
        if i != -1:
            # this is due to necessity to update all the spans after the deletion
            raise ValueError(
                "Removing messages other than the last one is not supported."
            )
        deleted_message = self.messages.pop(i)
        print("REMOVED MESSAGE", deleted_message)
        self.offset -= len(flatten(deleted_message["ids"]))
        if deleted_message["role"] == Source.assistant:
            self.supp_sent_spans = []

    def add_message(
        self,
        part: SamplePart | str,
        source: Union[Source.user, Source.assistant],
        ids: torch.Tensor | list[int] = None,
        wrapper: dict[str, dict] = None,
        to_continue: bool = False,
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
        :param to_continue: whether the model has to continue the last message (instead of answering to it)
        :return: None
        """
        if wrapper and ids is not None and source == Source.assistant:
            raise ValueError(
                "Wrapper can only be used for the messages created from scratch, and now, ids are passed."
            )
        self.part = part
        spans_types = {}
        # it is certainly a task
        if isinstance(part, SamplePart):
            if wrapper:
                print("DEBUG: case SamplePart and Wrapper")
                tokens, ids = [], []
                for key, wrap in wrapper.items():
                    intro, outro = wrap["before"], wrap.get("after", wrap["before"])
                    to_encode = None
                    type_ = "task"
                    if key == "context":
                        to_encode = part.structured_context.split("\n")
                        type_ = "cont"
                    elif key == "question":
                        to_encode = [part.structured_question]
                        type_ = "ques"

                    task_tokens, task_ids, task_spans = sents_to_ids(
                        to_encode, self.tokenizer
                    )
                    print("encoded message", *zip(task_spans, task_ids), sep="\n")

                    if task_ids:
                        ids_ = [intro["ids"], *task_ids, outro["ids"]]
                        tokens_ = [intro["tokens"], *task_tokens, outro["tokens"]]
                        for chunk_ids, chunk_tok in zip(ids_, tokens_):
                            if chunk_ids:
                                print(chunk_tok)
                                tokens.append(chunk_tok)
                                ids.append(chunk_ids)

                        # add before wrapper spans/ids
                        print("intro spans", intro["sent_spans"])
                        if intro["sent_spans"]:
                            spans_types[upd_span(intro["sent_spans"], self.offset)] = (
                                "wrap"
                            )
                            print(
                                "upd intro spans",
                                upd_span(intro["sent_spans"], self.offset),
                            )

                            self.offset += len(intro["ids"])

                        # add context and question spans/ids
                        for ids_, span in zip(task_ids, task_spans):
                            print("task span", span)
                            spans_types[upd_span(span, self.offset)] = type_
                            print("upd task spans", upd_span(span, self.offset))

                            self.offset += len(flatten(ids_))

                        # add after wrapper spans/ids
                        if outro["sent_spans"]:
                            print("outro spans", outro["sent_spans"])
                            spans_types[upd_span(outro["sent_spans"], self.offset)] = (
                                "wrap"
                            )
                            print(
                                "upd outro spans",
                                upd_span(outro["sent_spans"], self.offset),
                            )

                            self.offset += len(outro["ids"])
                    else:
                        # if the key is not context or question, we just add the before ids
                        # (no after because there's no formatting for reasoning nor answer)
                        ids.extend(intro["ids"])
                        tokens.extend(intro["tokens"])
                        spans_types[upd_span(intro["sent_spans"], self.offset)] = "wrap"

                        self.offset += len(intro["ids"])

            else:
                print("DEBUG: case SamplePart and NO Wrapper")
                tokens, ids, sent_spans = sents_to_ids(
                    part.unwrapped_task.split("\n"), self.tokenizer
                )
                spans_types.update(
                    {upd_span(span, self.offset): "task" for span in sent_spans}
                )
        else:
            # it is a string
            if ids is None:
                print("DEBUG: case str and NO ids")
                # it is a formatted prompt (string prompt) => task
                tokens, ids, sent_spans = sents_to_ids(part.split("\n"), self.tokenizer)
                spans_types.update(
                    {upd_span(span, self.offset): "teacher task" for span in sent_spans}
                )
            else:
                print("DEBUG: case str and ids")
                # it is certainly an assistant output
                # TODO: optionally divide it into reasoning and answer
                ids = ids.tolist() if not isinstance(ids, list) else ids
                tokens = self.tokenizer.convert_ids_to_tokens(ids)
                label = "ans" if source == Source.assistant else "task"
                spans_types[upd_span((0, len(ids)), self.offset)] = label
                self.offset += len(ids)

        print("ids", len(ids), ids)
        print("spans_types", spans_types)

        part_dict = {
            "role": source,
            "content": part if isinstance(part, str) else part.task,
            "original_content": part if isinstance(part, str) else part.unwrapped_task,
            "tokens": tokens,
            "ids": ids,
            "spans_types": spans_types,
        }
        self.messages.append(part_dict)

    def get_sentence_spans(
        self, span_type: str = "", remove_last: bool = False
    ) -> list[tuple[int, int]] | dict:
        """
        Get the all sentence spans of the chat messages for only a specified type of them.

        :param span_type: "sys" (system prompt),
                          "ex" (example),
                          "wrap" (wrappers),
                          "task" (context sentences and questions),
                          "cont" (context sentences),
                          "ques" (questions),
                          "ans" (model output)
        :param remove_last: whether to remove the last message span from the list
        :return: returns list of sentence spans if span type is specified otherwise returns all the spans with their types
        """
        possible_types = ("sys", "ex", "wrap", "task", "cont", "ques", "ans")
        if span_type and span_type not in possible_types:
            raise ValueError(
                f"Invalid span type: {span_type}. Valid types are: {', '.join(possible_types)}."
            )
        spans = []
        spans_dict = {}
        for message in self.messages[:-1] if remove_last else self.messages:
            # message["spans_types"] = {span: type}
            if span_type:
                for span, type_ in message["spans_types"].items():
                    if type_ == span_type or (
                        span_type == "task" and type_ in ["cont", "ques"]
                    ):
                        print("span", span_type, "type_", type_)
                        spans.append(span)
            else:
                spans_dict.update(message["spans_types"])

        if span_type:
            return spans
        return spans_dict

    def identify_supp_sent_spans(self):
        """
        Identify the supporting spans specific to the current part as they differ for each question.
        """
        self.supp_sent_spans = []
        all_task_spans = self.get_sentence_spans(span_type="task")
        for inx, span in enumerate(all_task_spans, 1):
            if inx in self.part.supporting_sent_inx:
                self.supp_sent_spans.append(span)

    def convert_into_ids(
        self, max_length: int = 2048, identify_target: bool = True
    ) -> torch.Tensor:
        """
        Converts the chat into ids using the tokenizer.

        :param max_length: the maximum length of the input
        :param identify_target: whether to identify the supporting sentence spans
        :return: list of ids
        """
        if identify_target:
            self.identify_supp_sent_spans()

        chat_ids = [self.tokenizer.convert_tokens_to_ids("<|begin_of_text|>")]
        # including the system prompt
        for i, message in enumerate(self.messages):
            message_ids = get_generation_tokens(self.tokenizer, message["role"])
            message_ids.extend(flatten(message["ids"]))
            conversation_length = len(chat_ids) + len(message_ids)
            if conversation_length > max_length:
                warnings.warn(
                    f"Exceeded max length with {conversation_length}. Truncating the input."
                )
                break

            chat_ids.extend(message_ids)

        chat_ids.extend(get_generation_tokens(self.tokenizer, "assistant"))
        print("chat_ids", len(chat_ids), chat_ids)

        return torch.as_tensor([chat_ids])
