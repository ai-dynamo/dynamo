# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build chat / Responses / Messages requests for a replay case and normalize replies.

Every endpoint's reply is reduced to one `Turn` (reasoning, content, tool calls,
finish reason, completion tokens) so replies can be compared with the trace and
with each other. Stream parsers also record protocol violations: events that a
client would mis-assemble even when the final text happens to be right.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

JsonDict = dict[str, Any]

ENDPOINTS: dict[str, str] = {
    "chat": "/v1/chat/completions",
    "responses": "/v1/responses",
    "messages": "/v1/messages",
}
# `max_tokens` is mandatory for the mocker; the replay script sets the real length.
_MAX_TOKENS_SLACK = 16
# Anthropic requires a thinking budget; replay ignores it.
_THINKING_BUDGET = 1024


@dataclass
class Turn:
    """One assistant reply, independent of the endpoint that produced it."""

    reasoning: str = ""
    content: str = ""
    tool_calls: list[JsonDict] = field(default_factory=list)  # {"name", "arguments"}
    finish_reason: str | None = None
    completion_tokens: int | None = None
    violations: list[str] = field(default_factory=list)
    events: int = 0


def replay_annotation(case: JsonDict) -> JsonDict:
    return {"annotations": [f"output_replay_id:{case['replay_key']}"]}


def build_request(
    endpoint: str, model: str, trajectory: JsonDict, case: JsonDict, stream: bool
) -> JsonDict:
    messages = trajectory["messages"][: case["message_index"]]
    if endpoint == "chat":
        return _chat_request(model, trajectory, messages, case, stream)
    if endpoint == "responses":
        return _responses_request(model, trajectory, messages, case, stream)
    if endpoint == "messages":
        return _messages_request(model, trajectory, messages, case, stream)
    raise ValueError(f"unknown endpoint {endpoint!r}")


def _chat_request(
    model: str,
    trajectory: JsonDict,
    messages: list[JsonDict],
    case: JsonDict,
    stream: bool,
) -> JsonDict:
    body: JsonDict = {
        "model": model,
        "messages": messages,
        "max_tokens": case["script_len"] + _MAX_TOKENS_SLACK,
        "stream": stream,
        "chat_template_args": trajectory["chat_template_args"],
        "nvext": replay_annotation(case),
    }
    if trajectory["tools"]:
        body["tools"] = trajectory["tools"]
    if stream:
        body["stream_options"] = {"include_usage": True}
    if case.get("stop"):
        body["stop"] = case["stop"]
    return body


def _responses_request(
    model: str,
    trajectory: JsonDict,
    messages: list[JsonDict],
    case: JsonDict,
    stream: bool,
) -> JsonDict:
    items: list[JsonDict] = []
    for position, message in enumerate(messages):
        role = message["role"]
        if role == "assistant":
            if message.get("reasoning_content"):
                items.append(
                    {
                        "type": "reasoning",
                        "id": f"rs_{position}",
                        "summary": [],
                        "content": [
                            {
                                "type": "reasoning_text",
                                "text": message["reasoning_content"],
                            }
                        ],
                    }
                )
            if message["content"]:
                items.append({"role": "assistant", "content": message["content"]})
            for call in message.get("tool_calls", []):
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
        else:
            items.append({"role": role, "content": message["content"]})
    body: JsonDict = {
        "model": model,
        "input": items,
        "max_output_tokens": case["script_len"] + _MAX_TOKENS_SLACK,
        "stream": stream,
        # Reasoning items are only returned when a summary is requested, as Codex does.
        "reasoning": {"summary": "auto"},
        "chat_template_args": trajectory["chat_template_args"],
        "nvext": replay_annotation(case),
    }
    if trajectory["tools"]:
        body["tools"] = [
            {"type": "function", **tool["function"]} for tool in trajectory["tools"]
        ]
    return body


def supports(endpoint: str, case: JsonDict) -> bool:
    """The Responses API has no stop sequences, so stop cases do not apply to it."""
    return not (endpoint == "responses" and case.get("stop"))


def _messages_request(
    model: str,
    trajectory: JsonDict,
    messages: list[JsonDict],
    case: JsonDict,
    stream: bool,
) -> JsonDict:
    system = [m["content"] for m in messages if m["role"] == "system"]
    turns: list[JsonDict] = []
    for message in messages:
        role = message["role"]
        if role == "assistant":
            blocks: list[JsonDict] = []
            if message.get("reasoning_content"):
                blocks.append(
                    {
                        "type": "thinking",
                        "thinking": message["reasoning_content"],
                        "signature": "",
                    }
                )
            if message["content"]:
                blocks.append({"type": "text", "text": message["content"]})
            for call in message.get("tool_calls", []):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": json.loads(call["function"]["arguments"]),
                    }
                )
            turns.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": message["content"],
            }
            # Anthropic answers all parallel calls in one user turn.
            if (
                turns
                and turns[-1]["role"] == "user"
                and _is_tool_result_turn(turns[-1])
            ):
                turns[-1]["content"].append(result)
            else:
                turns.append({"role": "user", "content": [result]})
        elif role == "user":
            turns.append({"role": "user", "content": message["content"]})
    thinking_on = trajectory["chat_template_args"].get(
        "enable_thinking",
        trajectory["chat_template_args"].get("thinking_mode") != "chat",
    )
    body: JsonDict = {
        "model": model,
        "messages": turns,
        "max_tokens": case["script_len"] + _MAX_TOKENS_SLACK,
        "stream": stream,
        "thinking": (
            {"type": "enabled", "budget_tokens": _THINKING_BUDGET}
            if thinking_on
            else {"type": "disabled"}
        ),
        "nvext": replay_annotation(case),
    }
    if system:
        body["system"] = "\n\n".join(system)
    if trajectory["tools"]:
        body["tools"] = [
            {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "input_schema": tool["function"].get("parameters", {"type": "object"}),
            }
            for tool in trajectory["tools"]
        ]
    if case.get("stop"):
        body["stop_sequences"] = case["stop"]
    return body


def _is_tool_result_turn(turn: JsonDict) -> bool:
    content = turn["content"]
    return isinstance(content, list) and all(
        block.get("type") == "tool_result" for block in content
    )


def parse_sse(lines: list[str]) -> list[tuple[str | None, str]]:
    """Split raw SSE lines into (event name, data) pairs."""
    events: list[tuple[str | None, str]] = []
    name: str | None = None
    data: list[str] = []
    for line in [*lines, ""]:
        if not line:
            if data:
                events.append((name, "\n".join(data)))
            name, data = None, []
        elif line.startswith("event:"):
            name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].lstrip())
    return events


def normalize(endpoint: str, stream: bool, payload: Any) -> Turn:
    """`payload` is the parsed JSON body (unary) or the raw SSE lines (stream)."""
    if endpoint == "chat":
        return _chat_stream(payload) if stream else _chat_unary(payload)
    if endpoint == "responses":
        return _responses_stream(payload) if stream else _responses_unary(payload)
    if endpoint == "messages":
        return _messages_stream(payload) if stream else _messages_unary(payload)
    raise ValueError(f"unknown endpoint {endpoint!r}")


def _chat_unary(body: JsonDict) -> Turn:
    choice = body["choices"][0]
    message = choice["message"]
    return Turn(
        reasoning=message.get("reasoning_content") or message.get("reasoning") or "",
        content=message.get("content") or "",
        tool_calls=[
            {
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            }
            for call in message.get("tool_calls") or []
        ],
        finish_reason=choice.get("finish_reason"),
        completion_tokens=(body.get("usage") or {}).get("completion_tokens"),
    )


def _chat_stream(lines: list[str]) -> Turn:
    turn = Turn()
    calls: dict[int, JsonDict] = {}
    roles = 0
    done = 0
    for _, data in parse_sse(lines):
        turn.events += 1
        if data == "[DONE]":
            done += 1
            continue
        if done:
            turn.violations.append("data-after-done")
        chunk = json.loads(data)
        if chunk.get("usage"):
            turn.completion_tokens = chunk["usage"].get("completion_tokens")
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            has_payload = any(
                delta.get(key) for key in ("content", "reasoning_content", "tool_calls")
            )
            if turn.finish_reason is not None and has_payload:
                turn.violations.append("delta-after-finish")
            if delta.get("role"):
                roles += 1
            turn.content += delta.get("content") or ""
            turn.reasoning += (
                delta.get("reasoning_content") or delta.get("reasoning") or ""
            )
            for call in delta.get("tool_calls") or []:
                index = call["index"]
                function = call.get("function") or {}
                if index not in calls:
                    if index != len(calls):
                        turn.violations.append("tool-index-gap")
                    if not call.get("id") or not function.get("name"):
                        turn.violations.append("tool-first-delta-missing-id-or-name")
                    calls[index] = {"name": function.get("name") or "", "arguments": ""}
                elif function.get("name"):
                    turn.violations.append("tool-name-repeated")
                calls[index]["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                if turn.finish_reason is not None:
                    turn.violations.append("multiple-finish-reasons")
                turn.finish_reason = choice["finish_reason"]
    if roles != 1:
        turn.violations.append(f"role-deltas={roles}")
    if done != 1:
        turn.violations.append(f"done-events={done}")
    turn.tool_calls = [calls[index] for index in sorted(calls)]
    return turn


def _responses_turn(response: JsonDict) -> Turn:
    turn = Turn()
    for item in response.get("output") or []:
        kind = item.get("type")
        if kind == "reasoning":
            turn.reasoning += "".join(
                part.get("text", "") for part in item.get("content") or []
            )
        elif kind == "message":
            turn.content += "".join(
                part.get("text", "")
                for part in item.get("content") or []
                if part.get("type") == "output_text"
            )
        elif kind == "function_call":
            turn.tool_calls.append(
                {"name": item["name"], "arguments": item["arguments"]}
            )
    status = response.get("status")
    if status == "completed":
        turn.finish_reason = "tool_calls" if turn.tool_calls else "stop"
    elif status == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        turn.finish_reason = (
            "length" if reason == "max_output_tokens" else f"incomplete:{reason}"
        )
    else:
        turn.finish_reason = status
    turn.completion_tokens = (response.get("usage") or {}).get("output_tokens")
    return turn


def _responses_unary(body: JsonDict) -> Turn:
    return _responses_turn(body)


def _responses_stream(lines: list[str]) -> Turn:
    deltas: dict[str, str] = {}
    final: JsonDict | None = None
    sequence = -1
    violations: list[str] = []
    events = 0
    for name, data in parse_sse(lines):
        events += 1
        if data == "[DONE]":
            # A chat-completions sentinel; Responses streams end at the terminal event.
            violations.append("done-sentinel")
            continue
        event = json.loads(data)
        kind = event.get("type")
        if name is not None and name != kind:
            violations.append("event-name-mismatch")
        if event.get("sequence_number", sequence + 1) <= sequence:
            violations.append("sequence-not-increasing")
        sequence = event.get("sequence_number", sequence)
        if kind in (
            "response.output_text.delta",
            "response.reasoning_text.delta",
            "response.function_call_arguments.delta",
        ):
            slot = f"{kind}:{event.get('output_index')}"
            deltas[slot] = deltas.get(slot, "") + event.get("delta", "")
        elif kind in ("response.completed", "response.incomplete", "response.failed"):
            if final is not None:
                violations.append("multiple-terminal-events")
            final = event["response"]
    if final is None:
        return Turn(violations=[*violations, "no-terminal-event"], events=events)
    turn = _responses_turn(final)
    turn.violations = violations
    turn.events = events
    # Streamed deltas must add up to the final output items.
    for index, item in enumerate(final.get("output") or []):
        kind = item.get("type")
        if kind == "function_call":
            streamed = deltas.get(f"response.function_call_arguments.delta:{index}")
            if streamed is not None and streamed != item["arguments"]:
                turn.violations.append("function-args-deltas-differ-from-final")
        elif kind == "message":
            text = "".join(part.get("text", "") for part in item.get("content") or [])
            if deltas.get(f"response.output_text.delta:{index}", "") != text:
                turn.violations.append("text-deltas-differ-from-final")
        elif kind == "reasoning":
            text = "".join(part.get("text", "") for part in item.get("content") or [])
            if deltas.get(f"response.reasoning_text.delta:{index}", "") != text:
                turn.violations.append("reasoning-deltas-differ-from-final")
    return turn


_ANTHROPIC_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}


def _messages_unary(body: JsonDict) -> Turn:
    turn = Turn()
    for block in body.get("content") or []:
        kind = block.get("type")
        if kind == "thinking":
            turn.reasoning += block.get("thinking", "")
        elif kind == "text":
            turn.content += block.get("text", "")
        elif kind == "tool_use":
            turn.tool_calls.append(
                {"name": block["name"], "arguments": json.dumps(block.get("input"))}
            )
    reason = body.get("stop_reason")
    turn.finish_reason = _ANTHROPIC_FINISH.get(reason, reason)
    turn.completion_tokens = (body.get("usage") or {}).get("output_tokens")
    return turn


def _messages_stream(lines: list[str]) -> Turn:
    turn = Turn()
    blocks: dict[int, JsonDict] = {}
    open_blocks: set[int] = set()
    stops = 0
    for name, data in parse_sse(lines):
        turn.events += 1
        if data == "[DONE]":
            # An OpenAI sentinel; the Anthropic protocol ends at message_stop.
            turn.violations.append("openai-done-sentinel")
            continue
        event = json.loads(data)
        kind = event.get("type")
        if name is not None and name != kind:
            turn.violations.append("event-name-mismatch")
        if stops:
            turn.violations.append("event-after-message-stop")
        if kind == "content_block_start":
            index = event["index"]
            if index in blocks:
                turn.violations.append("block-index-reused")
            if index != len(blocks):
                turn.violations.append("block-index-gap")
            blocks[index] = {**event["content_block"], "_json": ""}
            open_blocks.add(index)
        elif kind == "content_block_delta":
            index = event["index"]
            if index not in open_blocks:
                turn.violations.append("delta-for-closed-block")
                continue
            delta = event["delta"]
            block = blocks[index]
            if delta["type"] == "text_delta":
                block["text"] = block.get("text", "") + delta["text"]
            elif delta["type"] == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta["thinking"]
            elif delta["type"] == "input_json_delta":
                block["_json"] += delta["partial_json"]
        elif kind == "content_block_stop":
            if event["index"] not in open_blocks:
                turn.violations.append("stop-for-unopened-block")
            open_blocks.discard(event["index"])
        elif kind == "message_delta":
            reason = (event.get("delta") or {}).get("stop_reason")
            if reason:
                turn.finish_reason = _ANTHROPIC_FINISH.get(reason, reason)
            usage = event.get("usage") or {}
            if "output_tokens" in usage:
                turn.completion_tokens = usage["output_tokens"]
        elif kind == "message_stop":
            stops += 1
    if open_blocks:
        turn.violations.append("unclosed-blocks")
    if stops != 1:
        turn.violations.append(f"message-stop-events={stops}")
    for index in sorted(blocks):
        block = blocks[index]
        kind = block.get("type")
        if kind == "thinking":
            turn.reasoning += block.get("thinking", "")
        elif kind == "text":
            turn.content += block.get("text", "")
        elif kind == "tool_use":
            arguments = block["_json"] or json.dumps(block.get("input") or {})
            turn.tool_calls.append({"name": block["name"], "arguments": arguments})
    return turn
