# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble one chat stream into ordered realtime output items."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..connection import RealtimeTurn
from ..events import (
    conversation_item_added_event,
    conversation_item_done_event,
    event_id,
    response_created_event,
    response_done_event,
)
from ..request import TextGenerationRequest
from .text_utils import _realtime_usage


@dataclass
class _Output:
    kind: str
    item_id: str = field(default_factory=lambda: f"item_{uuid.uuid4().hex}")
    fragments: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    call_id: str | None = None
    announced: bool = False

    def item(self, status: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.item_id,
            "object": "realtime.item",
            "type": self.kind,
            "status": "completed" if status == "completed" else "incomplete",
        }
        if self.kind == "message":
            result.update(
                role="assistant",
                content=[{"type": "output_text", "text": "".join(self.fragments)}],
            )
        else:
            result.update(
                call_id=self.call_id,
                name="".join(self.names),
                arguments="".join(self.fragments),
            )
        return result


class TextTurn(RealtimeTurn):
    def __init__(
        self,
        *,
        messages: list[dict[str, Any]],
        max_output_tokens: int | None,
        wire_max_output_tokens: int | str,
        add_to_conversation: bool,
        items: list[dict[str, Any]],
        metadata: dict[str, str] | None = None,
        options: dict[str, Any] | None = None,
        emit_reasoning: bool = False,
    ) -> None:
        super().__init__()
        self.response_id = f"resp_{uuid.uuid4().hex}"
        self.messages = messages
        self.max_output_tokens = max_output_tokens
        self.request = TextGenerationRequest(messages, max_output_tokens, options or {})
        self.wire_max_output_tokens = wire_max_output_tokens
        self.add_to_conversation = add_to_conversation
        self.items = items
        self.metadata = metadata
        self.emit_reasoning = emit_reasoning
        self.previous_item_id = items[-1]["id"] if items else None
        self.outputs: list[_Output] = []
        self.tools: dict[int, _Output] = {}
        self.text_output: _Output | None = None
        self.reasoning: list[str] = []
        self.finished = False
        self.events.put_nowait(
            response_created_event(
                self.response_id,
                output_modalities=["text"],
                max_output_tokens=wire_max_output_tokens,
                metadata=metadata,
            )
        )

    def event(
        self, kind: str, output: _Output | None = None, **fields: Any
    ) -> dict[str, Any]:
        result = {
            "type": kind,
            "event_id": event_id(),
            "response_id": self.response_id,
            **fields,
        }
        if output is not None:
            result.update(
                item_id=output.item_id, output_index=self.outputs.index(output)
            )
        return result

    async def announce(self, output: _Output) -> None:
        if output.announced:
            return
        self.outputs.append(output)
        output.announced = True
        item = output.item("incomplete")
        item["status"] = "in_progress"
        if output.kind == "function_call":
            item["arguments"] = ""
        else:
            item["content"] = []
        await self.events.put(
            self.event("response.output_item.added", output, item=item)
        )
        if self.add_to_conversation:
            previous = self.previous_item_id
            index = self.outputs.index(output)
            if index:
                previous = self.outputs[index - 1].item_id
            await self.events.put(conversation_item_added_event(item, previous))
        if output.kind == "message":
            await self.events.put(
                self.event(
                    "response.content_part.added",
                    output,
                    content_index=0,
                    part={"type": "text", "text": ""},
                )
            )

    async def delta(self, delta: dict[str, Any]) -> None:
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, str):
                raise ValueError("reasoning must be text")
            if self.emit_reasoning and reasoning:
                self.reasoning.append(reasoning)
                await self.events.put(
                    self.event("response.nvext.reasoning.delta", delta=reasoning)
                )
        text = delta.get("content")
        if text is not None:
            if not isinstance(text, str):
                raise ValueError("chat completion returned non-text content")
            if text:
                if self.text_output is None:
                    self.text_output = _Output("message")
                output = self.text_output
                await self.announce(output)
                output.fragments.append(text)
                await self.events.put(
                    self.event(
                        "response.output_text.delta",
                        output,
                        content_index=0,
                        delta=text,
                    )
                )
        for call in delta.get("tool_calls") or []:
            if call.get("type") not in (None, "function"):
                raise ValueError("backend emitted a non-function tool call")
            index = call.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("tool call requires a nonnegative index")
            output = self.tools.get(index)
            if output is None:
                output = _Output("function_call")
                self.tools[index] = output
            call_id = call.get("id")
            if call_id is not None:
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("tool call requires a non-empty id")
                if output.call_id not in (None, call_id):
                    raise ValueError("tool call id changed during generation")
                if any(
                    other is not output and other.call_id == call_id
                    for other in self.tools.values()
                ):
                    raise ValueError("duplicate tool call id")
                output.call_id = call_id
            function = call.get("function") or {}
            name = function.get("name")
            if name is not None:
                if not isinstance(name, str):
                    raise ValueError("tool name must be a string")
                if name and output.announced:
                    if name != "".join(output.names):
                        raise ValueError("tool name changed during generation")
                elif name and name != "".join(output.names):
                    output.names.append(name)
            arguments = function.get("arguments")
            if arguments is not None:
                if not isinstance(arguments, str):
                    raise ValueError("tool arguments must be a string")
                output.fragments.append(arguments)
            names = {
                tool["function"]["name"]
                for tool in self.request.options.get("tools", [])
            }
            if output.call_id and "".join(output.names) in names and output.fragments:
                pending = (
                    "".join(output.fragments) if not output.announced else arguments
                )
                await self.announce(output)
                if pending:
                    await self.events.put(
                        self.event(
                            "response.function_call_arguments.delta",
                            output,
                            call_id=output.call_id,
                            delta=pending,
                        )
                    )

    def validate_completion(self, finish_reason: str | None) -> None:
        if finish_reason not in ("stop", "tool_calls", "length", "content_filter"):
            raise ValueError("chat stream ended without a supported finish reason")
        if finish_reason in ("length", "content_filter"):
            return
        names = {
            tool["function"]["name"] for tool in self.request.options.get("tools", [])
        }
        for output in self.tools.values():
            if not output.announced or "".join(output.names) not in names:
                raise ValueError("incomplete or unadvertised function call")
            if not isinstance(json.loads("".join(output.fragments)), dict):
                raise ValueError("function arguments must be a JSON object")
        choice = self.request.options.get("tool_choice", "none")
        if self.tools and choice == "none":
            raise ValueError("backend emitted a forbidden tool call")
        if (choice == "required" or isinstance(choice, dict)) and not self.tools:
            raise ValueError("backend did not emit the required tool call")
        if isinstance(choice, dict) and any(
            "".join(x.names) != choice["function"]["name"] for x in self.tools.values()
        ):
            raise ValueError("backend emitted a different named tool")
        if (
            self.request.options.get("parallel_tool_calls") is False
            and len(self.tools) > 1
        ):
            raise ValueError(
                "backend emitted multiple calls with parallel_tool_calls=false"
            )
        if finish_reason == "tool_calls" and not self.tools:
            raise ValueError("tool finish reason without a function call")

    def final_events(
        self,
        *,
        status: str,
        status_details: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.finished:
            return []
        self.finished = True
        events = []
        final_items = []
        previous = self.previous_item_id
        if self.reasoning:
            events.append(
                self.event(
                    "response.nvext.reasoning.done", text="".join(self.reasoning)
                )
            )
        for output in self.outputs:
            if not output.announced:
                continue
            item = output.item(status)
            if output.kind == "message":
                text = "".join(output.fragments)
                events.append(
                    self.event(
                        "response.output_text.done", output, content_index=0, text=text
                    )
                )
                events.append(
                    self.event(
                        "response.content_part.done",
                        output,
                        content_index=0,
                        part={"type": "text", "text": text},
                    )
                )
            else:
                events.append(
                    self.event(
                        "response.function_call_arguments.done",
                        output,
                        call_id=output.call_id,
                        name=item["name"],
                        arguments=item["arguments"],
                    )
                )
            events.append(self.event("response.output_item.done", output, item=item))
            final_items.append(item)
            if self.add_to_conversation:
                self.items.append(item)
                events.append(conversation_item_done_event(item, previous))
                previous = item["id"]
        done = response_done_event(
            self.response_id,
            output_modalities=["text"],
            max_output_tokens=self.wire_max_output_tokens,
            output=final_items,
            status=status,
            status_details=status_details,
            usage=_realtime_usage(usage),
            metadata=self.metadata,
        )
        if usage is not None:
            extra = {}
            reasoning = (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            )
            created = (usage.get("prompt_tokens_details") or {}).get(
                "created_cache_tokens"
            )
            if reasoning is not None:
                extra["reasoning_tokens"] = reasoning
            if created is not None:
                extra["created_cache_tokens"] = created
            if extra:
                done["response"]["nvext"] = {"usage": extra}
        events.append(done)
        return events
