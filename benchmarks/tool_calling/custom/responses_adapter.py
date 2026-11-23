# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Responses wire adaptation for the Custom qualification cases."""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Iterable, Iterator
from typing import Any


def _content_parts(content: Any, *, assistant: bool = False) -> Any:
    if not isinstance(content, list):
        return content
    converted = []
    for original in content:
        part = copy.deepcopy(original)
        if part.get("type") == "text":
            part["type"] = "output_text" if assistant else "input_text"
        elif part.get("type") == "image_url":
            image = part.pop("image_url")
            part.update(type="input_image", image_url=image["url"])
            if "detail" in image:
                part["detail"] = image["detail"]
        converted.append(part)
    return converted


def _function_identity(value: dict[str, Any]) -> dict[str, Any]:
    function = value.get("function", value)
    result = {"type": "function", "name": function["name"]}
    namespace = value.get("namespace", function.get("namespace"))
    if namespace is not None:
        result["namespace"] = namespace
    return result


def build_responses_payload(chat_payload: dict[str, Any]) -> dict[str, Any]:
    """Translate the shared Chat case payload without mutating its evidence."""
    payload = copy.deepcopy(chat_payload)
    messages = payload.pop("messages", [])
    input_items = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": _content_parts(message.get("content", "")),
                }
            )
            continue
        content = message.get("content")
        if content is not None and (content != "" or not message.get("tool_calls")):
            input_items.append(
                {
                    "role": role,
                    "content": _content_parts(content, assistant=role == "assistant"),
                }
            )
        for call in message.get("tool_calls") or []:
            function = call["function"]
            item = _function_identity(call)
            item.update(
                type="function_call",
                call_id=call["id"],
                arguments=function["arguments"],
            )
            input_items.append(item)
    payload["input"] = input_items
    if "tools" in payload:
        payload["tools"] = [
            {"type": "function", **tool["function"]}
            if tool.get("type") == "function" and "function" in tool
            else tool
            for tool in payload["tools"]
        ]
    choice = payload.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") == "allowed_tools":
            selection = choice.get("allowed_tools", choice)
            payload["tool_choice"] = {
                "type": "allowed_tools",
                "mode": selection["mode"],
                "tools": [_function_identity(tool) for tool in selection["tools"]],
            }
        elif choice.get("type") == "function":
            payload["tool_choice"] = _function_identity(choice)
    budget = payload.pop("max_completion_tokens", payload.pop("max_tokens", None))
    if budget is not None and "max_output_tokens" not in payload:
        payload["max_output_tokens"] = budget
    payload.pop("stream_options", None)
    if "reasoning_effort" in payload:
        effort = payload.pop("reasoning_effort")
        payload.setdefault("reasoning", {}).setdefault("effort", effort)
    response_format = payload.pop("response_format", None)
    if response_format:
        if response_format["type"] == "json_schema":
            response_format = {"type": "json_schema", **response_format["json_schema"]}
        payload["text"] = {"format": response_format}
    return payload


def _finding(kind: str, message: str, **details: Any) -> dict[str, Any]:
    return {"kind": kind, "message": message, "details": details}


def _usage(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    usage = {
        "prompt_tokens": raw.get("input_tokens"),
        "completion_tokens": raw.get("output_tokens"),
        "total_tokens": raw.get("total_tokens"),
    }
    for source, target in (
        ("input_tokens_details", "prompt_tokens_details"),
        ("output_tokens_details", "completion_tokens_details"),
    ):
        if source in raw:
            usage[target] = copy.deepcopy(raw[source])
    return usage


def parse_nonstream(raw: Any) -> dict[str, Any]:
    """Read native output while retaining status and call/item identity."""
    errors: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "api": "responses",
        "response_status": None,
        "raw_response": raw,
        "tool_calls": [],
        "content": "",
        "reasoning_content": "",
        "finish_reason": None,
        "parse_errors": errors,
    }
    if not isinstance(raw, dict):
        errors.append(_finding("responses_shape", "response is not an object"))
        return result
    result.update(
        response_id=raw.get("id") or "",
        model=raw.get("model") or "",
        created=raw.get("created_at"),
        response_status=raw.get("status"),
        usage=_usage(raw.get("usage")),
    )
    output = raw.get("output")
    if not isinstance(output, list):
        errors.append(_finding("responses_shape", "response.output is not a list"))
        output = []
    content: list[str] = []
    reasoning: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            errors.append(_finding("responses_shape", "output item is not an object"))
            continue
        if item.get("type") == "function_call":
            call = {
                "id": item.get("call_id") or "",
                "item_id": item.get("id"),
                "type": "function",
                "function": {
                    "name": item.get("name") or "",
                    "arguments": item.get("arguments"),
                },
            }
            if item.get("namespace") is not None:
                call["namespace"] = item["namespace"]
            if not isinstance(item.get("arguments"), str):
                errors.append(
                    _finding("responses_shape", "function arguments are not a string")
                )
            result["tool_calls"].append(call)
        elif item.get("type") in {"message", "reasoning"}:
            parts = item.get("content") or item.get("summary") or []
            if not isinstance(parts, list):
                errors.append(
                    _finding("responses_shape", "output content is not a list")
                )
                continue
            target = reasoning if item["type"] == "reasoning" else content
            for part in parts:
                if not isinstance(part, dict):
                    errors.append(
                        _finding("responses_shape", "content part is not an object")
                    )
                elif part.get("type") in {
                    "output_text",
                    "summary_text",
                    "reasoning_text",
                }:
                    if isinstance(part.get("text"), str):
                        target.append(part["text"])
                    else:
                        errors.append(
                            _finding("responses_shape", "output text is not a string")
                        )
    result["content"] = "".join(content)
    result["reasoning_content"] = "".join(reasoning)
    status = raw.get("status")
    if raw.get("error") is not None or status == "failed":
        errors.append(
            _finding(
                "responses_error", "Responses request failed", error=raw.get("error")
            )
        )
    elif status == "completed":
        result["finish_reason"] = "tool_calls" if result["tool_calls"] else "stop"
    elif status == "incomplete":
        details = raw.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        if reason == "max_output_tokens":
            result["finish_reason"] = "length"
        else:
            errors.append(
                _finding("responses_error", "response is incomplete", reason=reason)
            )
    else:
        errors.append(
            _finding(
                "responses_shape", "response has no terminal status", status=status
            )
        )
    return result


def _sse_events(lines: Iterable[bytes | str]) -> Iterator[tuple[str, str]]:
    event_name = ""
    data: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.rstrip("\r\n")
        if not line:
            if data:
                yield event_name, "\n".join(data)
            event_name, data = "", []
        elif line.startswith("event:"):
            event_name = line[6:].lstrip(" ")
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif line.startswith(":") or line.startswith(("id:", "retry:")):
            continue
        else:
            yield "invalid", line
    if data:
        yield event_name, "\n".join(data)


@dataclasses.dataclass
class _Item:
    snapshot: dict[str, Any] = dataclasses.field(default_factory=dict)
    arguments: list[str] = dataclasses.field(default_factory=list)
    arguments_done: str | None = None
    done: bool = False
    added: bool = False


def parse_stream(lines: Iterable[bytes | str]) -> dict[str, Any]:
    """Check event fragments against done and terminal snapshots, without repair."""
    errors: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    items: dict[int, _Item] = {}
    text: dict[tuple[int, int, str], list[str]] = {}
    text_done: dict[tuple[int, int, str], str] = {}
    terminal: dict[str, Any] | None = None

    def merge(index: int, snapshot: dict[str, Any]) -> _Item:
        state = items.setdefault(index, _Item())
        for key in ("id", "call_id", "name", "namespace", "type"):
            before, after = state.snapshot.get(key), snapshot.get(key)
            if before is not None and after is not None and before != after:
                errors.append(
                    _finding(
                        "responses_item_mismatch",
                        "output item identity changed",
                        output_index=index,
                        field=key,
                        before=before,
                        after=after,
                    )
                )
        state.snapshot.update(snapshot)
        return state

    for event_name, data in _sse_events(lines):
        if data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            records.append({"event": event_name, "data": data})
            errors.append(_finding("responses_shape", "invalid SSE JSON", data=data))
            continue
        records.append({"event": event_name, "data": event})
        if not isinstance(event, dict):
            errors.append(_finding("responses_shape", "SSE data is not an object"))
            continue
        kind = event.get("type") or event_name
        if not isinstance(kind, str):
            errors.append(_finding("responses_shape", "SSE event type is not a string"))
            continue
        if terminal is not None:
            errors.append(
                _finding(
                    "responses_event_mismatch",
                    "event received after terminal response",
                    event_type=kind,
                )
            )
        if event_name and event_name != kind:
            errors.append(
                _finding(
                    "responses_event_mismatch",
                    "SSE event and type disagree",
                    event=event_name,
                    payload_type=kind,
                )
            )
        if kind in {"error", "response.failed"}:
            errors.append(
                _finding(
                    "responses_error", "Responses stream reported an error", event=event
                )
            )
        if kind in {"response.completed", "response.incomplete", "response.failed"}:
            if terminal is not None:
                errors.append(
                    _finding(
                        "responses_duplicate_event", "multiple terminal response events"
                    )
                )
            response = event.get("response")
            if isinstance(response, dict):
                terminal = response
                expected_status = kind.split(".", 1)[1]
                if response.get("status") != expected_status:
                    errors.append(
                        _finding(
                            "responses_event_mismatch",
                            "terminal event and status disagree",
                        )
                    )
            else:
                errors.append(
                    _finding("responses_shape", "terminal event has no response object")
                )
            continue
        index = event.get("output_index")
        if index is not None and (
            not isinstance(index, int) or isinstance(index, bool) or index < 0
        ):
            errors.append(
                _finding("responses_shape", "invalid output index", output_index=index)
            )
            continue
        if kind in {"response.output_item.added", "response.output_item.done"}:
            snapshot = event.get("item")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not isinstance(snapshot, dict)
            ):
                errors.append(_finding("responses_shape", "invalid output item event"))
                continue
            state = merge(index, snapshot)
            if kind.endswith("added"):
                if state.added:
                    errors.append(
                        _finding(
                            "responses_duplicate_event",
                            "output item added twice",
                            output_index=index,
                        )
                    )
                state.added = True
            else:
                if state.done:
                    errors.append(
                        _finding(
                            "responses_duplicate_event",
                            "output item done twice",
                            output_index=index,
                        )
                    )
                state.done = True
                if snapshot.get("type") == "function_call":
                    expected = (
                        state.arguments_done
                        if state.arguments_done is not None
                        else "".join(state.arguments)
                    )
                    if (
                        state.arguments_done is not None or state.arguments
                    ) and snapshot.get("arguments") != expected:
                        errors.append(
                            _finding(
                                "responses_arguments_mismatch",
                                "item arguments differ from argument events",
                                output_index=index,
                            )
                        )
            continue
        if kind in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        }:
            if not isinstance(index, int) or isinstance(index, bool):
                errors.append(
                    _finding("responses_shape", "arguments event has no output index")
                )
                continue
            if not isinstance(event.get("item_id"), str) or not event["item_id"]:
                errors.append(
                    _finding("responses_shape", "arguments event has no item identity")
                )
                continue
            identity = {"id": event["item_id"], "type": "function_call"}
            if isinstance(event.get("name"), str):
                identity["name"] = event["name"]
            state = merge(index, identity)
            value_key = "delta" if kind.endswith("delta") else "arguments"
            value = event.get(value_key)
            if not isinstance(value, str):
                errors.append(
                    _finding(
                        "responses_shape",
                        "argument event is not a string",
                        output_index=index,
                    )
                )
                continue
            if kind.endswith("delta"):
                if state.arguments_done is not None or state.done:
                    errors.append(
                        _finding(
                            "responses_event_mismatch",
                            "arguments delta after completion",
                            output_index=index,
                        )
                    )
                state.arguments.append(value)
            else:
                if state.arguments_done is not None:
                    errors.append(
                        _finding(
                            "responses_duplicate_event",
                            "arguments done twice",
                            output_index=index,
                        )
                    )
                if state.arguments and "".join(state.arguments) != value:
                    errors.append(
                        _finding(
                            "responses_arguments_mismatch",
                            "argument deltas and done disagree",
                            output_index=index,
                        )
                    )
                state.arguments_done = value
            continue
        text_kinds = {
            "response.output_text": "text",
            "response.reasoning_text": "reasoning",
            "response.reasoning_summary_text": "summary",
        }
        text_kind, _, phase = kind.rpartition(".")
        if text_kind in text_kinds and phase in {"delta", "done"}:
            part_index = event.get("content_index", event.get("summary_index", 0))
            value = event.get("delta" if phase == "delta" else "text")
            if (
                not isinstance(index, int)
                or not isinstance(part_index, int)
                or isinstance(part_index, bool)
                or part_index < 0
                or not isinstance(value, str)
            ):
                errors.append(_finding("responses_shape", "invalid text event"))
                continue
            key = (index, part_index, text_kinds[text_kind])
            if phase == "delta":
                if key in text_done:
                    errors.append(
                        _finding(
                            "responses_event_mismatch",
                            "text delta after text completion",
                        )
                    )
                text.setdefault(key, []).append(value)
            else:
                if key in text_done:
                    errors.append(
                        _finding("responses_duplicate_event", "text completed twice")
                    )
                if key in text and "".join(text[key]) != value:
                    errors.append(
                        _finding(
                            "responses_event_mismatch", "text deltas and done disagree"
                        )
                    )
                text_done[key] = value

    if terminal is None:
        errors.append(
            _finding(
                "responses_stream_incomplete",
                "stream ended without a terminal response event",
            )
        )
        output = []
        for _, state in sorted(items.items()):
            snapshot = dict(state.snapshot)
            if state.arguments:
                snapshot["arguments"] = "".join(state.arguments)
            output.append(snapshot)
        result = parse_nonstream({"output": output, "status": None})
    else:
        result = parse_nonstream(terminal)
        output = terminal.get("output") or []
        if isinstance(output, list):
            for index, snapshot in enumerate(output):
                if not isinstance(snapshot, dict):
                    continue
                existing = items.get(index)
                state = merge(index, snapshot)
                if (
                    terminal.get("status") == "completed"
                    and snapshot.get("type") == "function_call"
                ):
                    if (
                        existing is None
                        or not state.added
                        or not state.done
                        or state.arguments_done is None
                    ):
                        errors.append(
                            _finding(
                                "responses_stream_incomplete",
                                "completed function call lacks lifecycle events",
                                output_index=index,
                            )
                        )
                    if (
                        state.arguments_done is not None
                        and state.arguments_done != snapshot.get("arguments")
                    ):
                        errors.append(
                            _finding(
                                "responses_arguments_mismatch",
                                "terminal and streamed arguments disagree",
                                output_index=index,
                            )
                        )
                for field, channel in (
                    (
                        "content",
                        "reasoning" if snapshot.get("type") == "reasoning" else "text",
                    ),
                    ("summary", "summary"),
                ):
                    parts = snapshot.get(field) or []
                    if not isinstance(parts, list):
                        continue
                    for part_index, part in enumerate(parts):
                        if not isinstance(part, dict) or part.get("type") not in {
                            "output_text",
                            "reasoning_text",
                            "summary_text",
                        }:
                            continue
                        key = (index, part_index, channel)
                        pieces = text.get(key)
                        if (
                            terminal.get("status") == "completed"
                            and snapshot.get("type") == "message"
                            and part.get("type") == "output_text"
                            and part.get("text")
                            and (
                                not state.added
                                or not state.done
                                or pieces is None
                                or key not in text_done
                            )
                        ):
                            errors.append(
                                _finding(
                                    "responses_stream_incomplete",
                                    "completed text lacks lifecycle events",
                                    output_index=index,
                                    content_index=part_index,
                                )
                            )
                        if (
                            pieces is not None and "".join(pieces) != part.get("text")
                        ) or (key in text_done and text_done[key] != part.get("text")):
                            errors.append(
                                _finding(
                                    "responses_event_mismatch",
                                    "terminal text and streamed text disagree",
                                    output_index=index,
                                    content_index=part_index,
                                )
                            )
            if terminal.get("status") == "completed" and len(output) != len(items):
                errors.append(
                    _finding(
                        "responses_item_mismatch",
                        "terminal output omits streamed items",
                    )
                )
    result["raw_response"] = records
    result["chunk_count"] = len(records)
    result["parse_errors"].extend(errors)
    return result
