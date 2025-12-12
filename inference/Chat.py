from __future__ import annotations

import copy
import warnings
from typing import Union

import torch
from transformers import PreTrainedTokenizerFast

from inference.DataLevels import SamplePart
from inference.Prompt import Prompt
from inference.utils import (
    Source,
    flatten,
    get_generation_token_ids,
    sents_to_ids,
    type_is_task,
    update_span,
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

        sys_prompt_spans_with_types = {
            span: "sys" for span in system_prompt.orig_sent_spans
        }
        example_spans_with_types = {span: "ex" for span in system_prompt.ex_sent_spans}

        print("system_prompt.ids", system_prompt.ids)

        self.system_message = {
            "role": Source.system,
            "content": system_prompt.text,
            "original_content": system_prompt.original_text,
            "tokens": system_prompt.tokens,
            "ids": system_prompt.ids,
            "spans_with_types": {
                **sys_prompt_spans_with_types,
                **example_spans_with_types,
            },
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

    def adjust_message(
        self,
        output: str,
        ids: int | list[int] | torch.Tensor,
        full_output: bool = False,
    ) -> None:
        """
        Adjust the *last* message of the chat.
        This is mostly used in SD to update the last message with the partial output of the model.

        :param output: the partial output of the model
        :param ids: the ids of the partial output
        :param full_output: whether the output is the full output or just a partial one
        :return: None
        """
        if not self.messages:
            raise ValueError("No messages to adjust.")

<<<<<<< HEAD
        if type(partial_ids) == torch.Tensor:
            partial_ids = partial_ids.tolist()
        elif type(partial_ids) == int:
            partial_ids = [partial_ids]
=======
        if type(ids) is torch.Tensor:
            ids = ids.tolist()
        if type(ids) is int:
            ids = [ids]
>>>>>>> fdbf27c13f1d77bbff65723971ae2f88be973c3e
        else:
            ids = flatten(ids)

<<<<<<< HEAD
=======
        assert type(ids[0]) is int

>>>>>>> fdbf27c13f1d77bbff65723971ae2f88be973c3e
        if type(self.messages[-1]["tokens"]) is str:
            raise ValueError(
                "Detected tokens instead of token lists. Please check the input."
            )

<<<<<<< HEAD
        self.messages[-1]["content"] += partial_output
        self.messages[-1]["original_content"] += partial_output
        print("self.messages[-1]", self.messages[-1])
        print(
            "self.tokenizer.convert_ids_to_tokens(partial_ids)",
            self.tokenizer.convert_ids_to_tokens(partial_ids),
        )
        self.messages[-1]["tokens"][-1].extend(
            self.tokenizer.convert_ids_to_tokens(partial_ids)
        )
        self.messages[-1]["ids"][-1].extend(partial_ids)
=======
        if not full_output:
            self.messages[-1]["content"] += output
            self.messages[-1]["original_content"] += output
            self.messages[-1]["tokens"][-1].extend(
                self.tokenizer.convert_ids_to_tokens(ids)
            )
            self.messages[-1]["ids"][-1].extend(ids)
            self.offset += len(ids)
        else:
            self.messages[-1]["content"] = output
            self.messages[-1]["original_content"] = output
            self.messages[-1]["tokens"][-1] = self.tokenizer.convert_ids_to_tokens(ids)
            self.messages[-1]["ids"][-1] = ids
            self.offset = len(ids)

>>>>>>> fdbf27c13f1d77bbff65723971ae2f88be973c3e
        spans_with_types = self.messages[-1]["spans_with_types"]
        model_output_span = list(spans_with_types.keys())[-1]
        model_output_span_type = {
            (model_output_span[0], model_output_span[1] + len(ids)): "ans"
        }
        self.messages[-1]["spans_with_types"] = {
            **dict(list(spans_with_types.items())[:-1]),
            **model_output_span_type,
        }

    def add_message(
        self,
        part: SamplePart | str | dict,
        source: Union[Source.user, Source.assistant] = None,
        ids: torch.Tensor | list[int] = None,
        tokens: list[str] = None,
        wrapper: dict[str, dict] = None,
        **kwargs,
    ) -> None:
        """
        Add a message to the messages list. It can addition either a message from the part task or assistant output.
        The message is converted into ids using the tokenizer, but the ids can also be provided for the assistant
        output.
        If the wrapper is present, it takes into account its ids and spans to offset the sentence spans and adds it
        to the task ids.

        :param part: part of a sample as a string or a list of strings
        :param source: the producer of the message
        :param ids: the ids of the message, if None, the ids are generated from the message
        :param tokens: the tokens of the message, if None, the tokens are generated from the message
        :param wrapper: the wrapper ids and sentence spans of the message
        :return: None
        """
        if kwargs:
            warnings.warn("Unused keyword arguments: " + ", ".join(kwargs.keys()))
        if (
            wrapper
            and (ids is not None or type(part) in [str, dict])
            and source == Source.assistant
        ):
            raise ValueError(
                "Wrapper can only be used for the messages created from scratch, and now, ids are passed."
            )

        message_fields = (
            "role",
            "content",
            "original_content",
            "tokens",
            "ids",
            "spans_with_types",
        )
        self.part = part
        spans_with_types = {}
        part_dict = {}
        # it is a pre-created message (used in SD and Feedback)
        if isinstance(part, dict):
            if all(key in part for key in message_fields):
                print("DEBUG: case dict and all keys present")
                part["spans_with_types"] = {
                    update_span(span, self.offset): f"{type_}_"
                    for span, type_ in part["spans_with_types"].items()
                }
                self.offset += len(flatten(["ids"]))
                part_dict = part
            else:
                raise ValueError(
                    f"The part is a dictionary, but not all required fields are present: {part}"
                )
        # it is certainly a task
        elif isinstance(part, SamplePart):
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
                    task_chunks = [
                        {
                            "tokens": tokens,
                            "ids": ids,
                            "sent_spans": spans,
                            "type": type_,
                        }
                        for tokens, ids, spans in zip(task_tokens, task_ids, task_spans)
                    ]
                    print("encoded message", *zip(task_spans, task_ids), sep="\n")
                    chunks = [intro, *task_chunks, outro]
                    last_span = ()
                    for i, chunk in enumerate(chunks):
                        if chunk.get("ids", None):
                            tokens.append(chunk["tokens"])
                            ids.append(chunk["ids"])
                            type_ = chunk["type"] if chunk.get("type", None) else "wrap"
                            upd_span = update_span(chunk["sent_spans"], self.offset)

                            if last_span and last_span[1] != upd_span[0]:
                                print("DEBUG: chunks", *chunks)
                                raise ValueError(
                                    f"Span mismatch: {last_span} vs {upd_span}"
                                )
                            last_span = upd_span
                            spans_with_types[upd_span] = type_
                            flat_chunk_ids = flatten(chunk["ids"])
                            self.offset += len(flat_chunk_ids)

                            if len(flat_chunk_ids) != upd_span[1] - upd_span[0]:
                                print("DEBUG: chunks", *chunks)
                                print("flat_chunk_ids", flat_chunk_ids)
                                print("upd_span difference", upd_span[1] - upd_span[0])
                                raise ValueError(
                                    f"Span length mismatch: {flatten(chunk['ids'])}, {chunk['sent_spans']} vs {upd_span}"
                                )

                        else:
                            print("Chat ids is none")
                            print("DEBUG: chunks", *chunks)

            else:
                print("DEBUG: case SamplePart and NO Wrapper")
                tokens, ids, sent_spans = sents_to_ids(
                    part.unwrapped_task.split("\n"), self.tokenizer
                )
                spans_with_types.update(
                    {update_span(span, self.offset): "task" for span in sent_spans}
                )
        else:
            # it is a string
            if ids is None:
                print("DEBUG: case str and NO ids")
                # it is a formatted prompt (string prompt) => task
                tokens, ids, sent_spans = sents_to_ids(part.split("\n"), self.tokenizer)
                spans_with_types.update(
                    {
                        update_span(span, self.offset): "teacher task"
                        for span in sent_spans
                    }
                )
            else:
                print("DEBUG: case str and ids")
                # it is certainly an assistant output
                # TODO: optionally divide it into reasoning and answer
                ids = ids.tolist() if not isinstance(ids, list) else ids
                # not flat because they count as "one sentence"
                if type(ids[0]) is int:
                    ids = [ids]
                if not tokens:
                    tokens = [
                        self.tokenizer.convert_ids_to_tokens(id_list) for id_list in ids
                    ]
                elif type(tokens[0]) is str:
                    tokens = [tokens]
                label = "ans" if source == Source.assistant else "task"
                spans_with_types[update_span((0, len(ids)), self.offset)] = label
                self.offset += len(ids)

        part_dict = part_dict or {
            "role": source,
            "content": part if isinstance(part, str) else part.task,
            "original_content": part if isinstance(part, str) else part.unwrapped_task,
            "tokens": tokens,
            "ids": ids,
            "spans_with_types": spans_with_types,
        }
        print("ids", len(part_dict["ids"]), part_dict["ids"])
        print("spans_with_types", part_dict["spans_with_types"])
        self.messages.append(part_dict)

    def move_approved_message(
        self, other_chat: Chat, wrapper: dict = None, source: str = ""
    ) -> None:
        """
        Move the last message from another chat to the current chat.

        :param other_chat: the chat to move the message from
        :param wrapper: the wrapper for the moved message
        :param source: the source of the message, if not provided, it will be taken from the other chat
        :return: None
        """
<<<<<<< HEAD
        approved_message = other_chat.messages[-1]
        print("MOVING MESSAGE", approved_message, sep="\n")
=======
        approved_message = copy.deepcopy(other_chat.messages[-1])
>>>>>>> fdbf27c13f1d77bbff65723971ae2f88be973c3e
        offset_difference = other_chat.offset - self.offset
        spans_with_types = {}

        if "The student's response was:" in approved_message["original_content"]:
            raise ValueError(
                f"The message contains teacher wrapper: {approved_message}"
            )

        st_resp_wrapper = ["The", "Ġstudent", "'s", "Ġresponse", "Ġwas", ":Ċ"]
        content, original_content = "", ""
        message_ids, message_tokens = [], []
        if wrapper:
            for key, wrap in wrapper.items():
                intro, outro = wrap["before"], wrap.get("after", wrap["before"])
                chunks = [intro, approved_message, outro]
                for chunk in chunks:
                    assert type(chunk) is dict
                    content += chunk.get("original_content", chunk["content"])
                    original_content += chunk.get("original_content", "")
                    if type(chunk["ids"]) is int:
                        message_ids.append(chunk["ids"])
                        message_tokens.append(chunk["tokens"])
                    else:
                        for ids, tokens in zip(chunk["ids"], chunk["tokens"]):
                            if st_resp_wrapper == tokens:
                                continue
                            message_ids.append(ids)
                            message_tokens.append(tokens)

                    updated_span = update_span(chunk["sent_spans"], self.offset)
                    spans_with_types[updated_span] = chunk.get("type", "wrap")
                    self.offset += len(flatten(chunk["ids"]))
            approved_message = {
                "role": source or approved_message["role"],
                "content": content,
                "original_content": original_content,
                "tokens": message_tokens,
                "ids": message_ids,
                "spans_with_types": spans_with_types,
            }
        else:
            for span, type_ in approved_message["spans_with_types"].items():
                spans_upd = (span[0] - offset_difference, span[1] - offset_difference)
                spans_with_types[spans_upd] = type_
            approved_message["spans_with_types"] = spans_with_types
            approved_message["content"] = approved_message["original_content"]
        self.messages.append(approved_message)

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
        :return: returns list of sentence spans if span type is specified otherwise returns all the spans with their
        types
        """
        possible_types = ("sys", "ex", "wrap", "task", "cont", "ques", "ans")
        if span_type and span_type not in possible_types:
            raise ValueError(
                f"Invalid span type: {span_type}. Valid types are: {', '.join(possible_types)}."
            )
        spans = []
        spans_dict = {}
        for message in self.messages[:-1] if remove_last else self.messages:
<<<<<<< HEAD
            # message["spans_with_types"] = {span: type}
            if span_type:
                for span, type_ in message["spans_with_types"].items():
                    if type_ == span_type or (
                        span_type == "task" and type_ in ["cont", "ques"]
                    ):
                        print("span", span_type, "type_", type_)
=======
            for span, type_ in message["spans_with_types"].items():
                if span_type:
                    if type_ == span_type or type_is_task(span_type, type_):
>>>>>>> fdbf27c13f1d77bbff65723971ae2f88be973c3e
                        spans.append(span)
                else:
                    spans_dict[span] = type_

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
        if len(self.supp_sent_spans) != len(self.part.supporting_sent_inx):
            warnings.warn(
                f"Number of supporting sentence spans ({len(self.supp_sent_spans)}) "
                f"does not match the number of supporting sentences in the part "
                f"({len(self.part.supporting_sent_inx)})."
            )

    def convert_into_datatype(
        self,
        datatype: str,
        max_length: int = 8000,
        identify_target: bool = True,
        to_continue: bool = False,
    ) -> torch.Tensor:
        """
        Converts the chat into 'ids' or 'tokens' using the tokenizer.

        :param datatype: what to convert chat into ('ids' or 'tokens')
        :param max_length: the maximum length of the input
        :param identify_target: whether to identify the supporting sentence spans
        :param to_continue: whether the last message has to be continued (no generation token will be added)
        :return: list of ids or tokens as a tensor
        """
        if datatype not in ("ids", "tokens"):
            raise ValueError(
                f"Value {datatype} is not supported. Must be either 'ids' or 'tokens'."
            )

        chat_tokens = []
        chat_ids = []
        conversation_length = len(chat_ids)
        # including the system prompt
        for i, message in enumerate(self.messages):
<<<<<<< HEAD
            message_ids = get_generation_token_ids(self.tokenizer, message["role"])
            if type(message[datatype][0]) is str:
                raise ValueError(
                    "Detected tokens instead of ids. Please check the input."
                )
=======
            message_ids, message_tokens = get_generation_token_ids(
                self.tokenizer, message["role"], start=i == 0
            )
>>>>>>> fdbf27c13f1d77bbff65723971ae2f88be973c3e
            message_ids.extend(flatten(message[datatype]))
            message_tokens.extend(flatten(message["tokens"]))
            conversation_length += len(message_ids)
            if conversation_length > max_length:
                warnings.warn(
                    f"Exceeded max length {conversation_length}. Truncating the input."
                )
                break

            chat_ids.extend(message_ids)
            chat_tokens.extend(message_tokens)

            print(f"DEBUG: chat_ids as str: {self.tokenizer.decode(chat_ids)}")
        if not to_continue:
<<<<<<< HEAD
            chat_ids.extend(get_generation_token_ids(self.tokenizer, "assistant"))
        print("chat_ids", len(chat_ids), chat_ids)
=======
            gen_ids, _ = get_generation_token_ids(self.tokenizer, "assistant")
            chat_ids.extend(gen_ids)

        if identify_target:
            self.identify_supp_sent_spans()
>>>>>>> fdbf27c13f1d77bbff65723971ae2f88be973c3e

        return torch.as_tensor([chat_ids])
