# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native GLM output parsing with bounded Dynamo recovery exceptions."""

import json
import re
from typing import Any

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import Unresolvable
from sglang.srt.function_call.function_call_parser import FunctionCallParser


class GlmOutputParser:
    def __init__(self, *, reasoning_parser, tool_parser, tools, stream: bool):
        self.reasoning_parser = reasoning_parser
        self.tool_parser = tool_parser
        self.tools = {tool.function.name: tool for tool in tools or []}
        self.stream = stream
        self._text: list[str] = []
        self._tool_text: list[str] = []
        self._names: dict[int, str] = {}
        self._arguments: dict[int, list[str]] = {}
        self._indices: dict[int, int] = {}
        self.has_calls = False

    def _calls(self, calls) -> list[dict[str, Any]]:
        result = []
        for call in calls:
            index = call.tool_index
            if call.name:
                self._names[index] = call.name
            name = self._names.get(index)
            if not name or name not in self.tools:
                continue
            if index not in self._indices:
                self._indices[index] = len(self._indices)
            self._arguments.setdefault(index, []).append(call.parameters or "")
            function: dict[str, Any] = {"arguments": call.parameters or ""}
            if call.name:
                function["name"] = call.name
            result.append({"index": self._indices[index], "function": function})
        self.has_calls |= bool(result)
        return result

    def _has_malformed_arguments(self, indices: set[int] | None = None) -> bool:
        arguments = (
            self._arguments.values()
            if indices is None
            else (
                self._arguments[index] for index in indices if index in self._arguments
            )
        )
        for parts in arguments:
            try:
                json.loads("".join(parts))
            except ValueError:
                return True
        return False

    def _recover_later_calls(self) -> list[dict[str, Any]]:
        # Never rewrite arguments that have already reached the client.
        if not self._has_malformed_arguments() or not self.tool_parser:
            return []
        text = "".join(self._tool_text)
        frames = list(re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL))
        end = 0
        for frame in frames:
            if "<tool_call>" in frame.group(1) or any(
                marker in text[end : frame.start()]
                for marker in ("</arg_value>", "</arg_key>", "</tool_call>")
            ):
                return []
            end = frame.end()
        if any(
            marker in text[end:]
            for marker in ("</arg_value>", "</arg_key>", "</tool_call>")
        ):
            return []
        parser = FunctionCallParser(
            tools=list(self.tools.values()), tool_call_parser="glm47"
        )
        _, calls = parser.parse_non_stream(text)
        if len(calls) != len(frames):
            return []
        highest_index = max(self._names, default=-1)
        recovered = []
        validators = {}
        for ordinal, call in enumerate(calls):
            if ordinal <= highest_index or call.name not in self.tools:
                continue
            name, separator, body = frames[ordinal].group(1).partition("<arg_key>")
            if not separator or name.strip() != call.name:
                continue
            body = "<arg_key>" + body
            fields = list(
                re.finditer(
                    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                    body,
                    re.DOTALL,
                )
            )
            end = 0
            keys = set()
            complete = bool(fields)
            for field in fields:
                key = field.group(1).strip()
                if (
                    body[end : field.start()].strip()
                    or key in keys
                    or any(
                        marker in field.group(1) or marker in field.group(2)
                        for marker in (
                            "<arg_key>",
                            "</arg_key>",
                            "<arg_value>",
                            "</arg_value>",
                        )
                    )
                ):
                    complete = False
                    break
                keys.add(key)
                end = field.end()
            if not complete or body[end:].strip():
                continue
            schema = self.tools[call.name].function.parameters or {}
            try:
                arguments = json.loads(call.parameters)
                json.dumps(arguments, allow_nan=False)
                if call.name not in validators:
                    validator = validator_for(schema)
                    validator.check_schema(schema)
                    validators[call.name] = validator(schema, registry=Registry())
                valid = (
                    isinstance(arguments, dict)
                    and bool(arguments)
                    and validators[call.name].is_valid(arguments)
                )
            except (ValueError, SchemaError, Unresolvable, RecursionError):
                continue
            if valid:
                # Batch indices refer to tool definitions, not call ordinals.
                call.tool_index = ordinal
                recovered.append(call)
        return self._calls(recovered)

    def parse(self, text: str, finish_reason: str | None) -> list[dict[str, Any]]:
        if not self.stream:
            self._text.append(text)
            if not finish_reason:
                return []
            text = "".join(self._text)
            reasoning = ""
            if self.reasoning_parser:
                reasoning, text = self.reasoning_parser.parse_non_stream(text)
            calls = []
            if self.tool_parser and self.tool_parser.has_tool_call(text or ""):
                text, calls = self.tool_parser.parse_non_stream(text)
                for index, call in enumerate(calls):
                    call.tool_index = index
            delta: dict[str, Any] = {}
            if reasoning:
                delta["reasoning_content"] = reasoning
            if text:
                delta["content"] = text
            if projected := self._calls(calls):
                delta["tool_calls"] = projected
            return [delta] if delta else []

        reasoning = ""
        if self.reasoning_parser:
            reasoning, text = self.reasoning_parser.parse_stream_chunk(text)
            if finish_reason and (
                flush := getattr(self.reasoning_parser, "parse_stream_end", None)
            ):
                last_reasoning, last_text = flush()
                reasoning = (reasoning or "") + (last_reasoning or "")
                text = (text or "") + (last_text or "")
        calls = []
        if self.tool_parser and (text or finish_reason):
            self._tool_text.append(text or "")
            text, calls = self.tool_parser.parse_stream_chunk(text or "")
        delta = {}
        if reasoning:
            delta["reasoning_content"] = reasoning
        if text:
            delta["content"] = text
        if projected := self._calls(calls):
            delta["tool_calls"] = projected
        deltas = [delta] if delta else []
        if finish_reason and self.tool_parser:
            malformed = self._has_malformed_arguments()
            # Preserve each detector result's text-before-calls order.
            for _ in range("".join(self._tool_text).count("</tool_call>") + 1):
                if malformed:
                    break
                last_text, last_calls = self.tool_parser.parse_stream_chunk("")
                if not last_text and not last_calls:
                    break
                delta = {}
                if last_text:
                    delta["content"] = last_text
                if projected := self._calls(last_calls):
                    delta["tool_calls"] = projected
                if delta:
                    deltas.append(delta)
                malformed = self._has_malformed_arguments(
                    {call.tool_index for call in last_calls}
                )
            if flush := getattr(self.tool_parser, "parse_stream_end", None):
                last_text, last_calls = flush()
                delta = {}
                if last_text:
                    delta["content"] = last_text
                if projected := self._calls(last_calls):
                    delta["tool_calls"] = projected
                if delta:
                    deltas.append(delta)
        if finish_reason and (recovered := self._recover_later_calls()):
            deltas.append({"tool_calls": recovered})
        return deltas
