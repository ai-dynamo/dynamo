# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Realtime text options and normalize conversation items and usage."""

from __future__ import annotations

import uuid
from typing import Any

TOOL_CHOICE_MODES = ("auto", "none", "required")
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


def _max_output_tokens(value: Any) -> tuple[int | None, int | str]:
    if value in (None, "inf"):
        return None, "inf"
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4096:
        raise ValueError("max_output_tokens must be an integer from 1 to 4096 or 'inf'")
    return value, value


def _validate_tools(tools: Any) -> None:
    if tools is None:
        return
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    for tool in tools:
        if (
            not isinstance(tool, dict)
            or tool.get("type") != "function"
            or not isinstance(tool.get("name"), str)
            or not tool["name"]
        ):
            raise ValueError("each tool must be a function with a name")
        if not isinstance(tool.get("parameters", {}), dict):
            raise ValueError(f"tool {tool['name']!r} parameters must be an object")


def _validate_tool_choice(tool_choice: Any) -> None:
    if tool_choice is None or tool_choice in TOOL_CHOICE_MODES:
        return
    if (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and isinstance(tool_choice.get("name"), str)
    ):
        return
    raise ValueError("tool_choice must be auto, none, required, or a named function")


def _validate_text_options(options: dict[str, Any], name: str) -> None:
    if options.get("output_modalities") not in (None, ["text"]):
        raise ValueError("only text output is supported")
    _validate_tools(options.get("tools"))
    _validate_tool_choice(options.get("tool_choice"))
    if not isinstance(options.get("parallel_tool_calls", False), bool):
        raise ValueError(f"{name}.parallel_tool_calls must be a boolean")
    _reasoning_options(options.get("reasoning"))
    if options.get("prompt") is not None:
        raise ValueError("prompt configuration is not supported")
    if not isinstance(options.get("instructions", ""), str):
        raise ValueError(f"{name}.instructions must be a string")


def _item_id(value: dict[str, Any]) -> str:
    item_id = value.get("id") or f"item_{uuid.uuid4().hex}"
    if not isinstance(item_id, str):
        raise ValueError("conversation item id must be a string")
    return item_id


def _normalize_function_call_item(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value.get("call_id"), str) or not value["call_id"]:
        raise ValueError("function_call call_id must be a non-empty string")
    if not isinstance(value.get("name"), str) or not value["name"]:
        raise ValueError("function_call name must be a non-empty string")
    arguments = value.get("arguments", "")
    if not isinstance(arguments, str):
        raise ValueError("function_call arguments must be a JSON string")
    return {
        "id": _item_id(value),
        "object": "realtime.item",
        "type": "function_call",
        "status": "completed",
        "call_id": value["call_id"],
        "name": value["name"],
        "arguments": arguments,
    }


def _normalize_function_call_output_item(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value.get("call_id"), str) or not value["call_id"]:
        raise ValueError("function_call_output call_id must be a non-empty string")
    if not isinstance(value.get("output"), str):
        raise ValueError("function_call_output output must be a string")
    return {
        "id": _item_id(value),
        "object": "realtime.item",
        "type": "function_call_output",
        "status": "completed",
        "call_id": value["call_id"],
        "output": value["output"],
    }


def _normalize_text_item(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("type") == "function_call":
        return _normalize_function_call_item(value)
    if isinstance(value, dict) and value.get("type") == "function_call_output":
        return _normalize_function_call_output_item(value)
    if not isinstance(value, dict) or value.get("type") != "message":
        raise ValueError(
            "only message, function_call, and function_call_output items are supported"
        )
    role = value.get("role")
    if role not in {"system", "user", "assistant"}:
        raise ValueError("message role must be system, user, or assistant")

    content_type = "output_text" if role == "assistant" else "input_text"
    content = value.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("message content must be a non-empty array")
    if any(
        not isinstance(part, dict)
        or part.get("type") != content_type
        or not isinstance(part.get("text"), str)
        for part in content
    ):
        raise ValueError(f"{role} message content must contain only {content_type}")

    return {
        "id": _item_id(value),
        "object": "realtime.item",
        "type": "message",
        "status": "completed",
        "role": role,
        "content": [{"type": content_type, "text": part["text"]} for part in content],
    }


def _tool_call(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["call_id"],
        "type": "function",
        "function": {"name": item["name"], "arguments": item["arguments"]},
    }


def _text_prompt(
    items: list[dict[str, Any]], instructions: str
) -> list[dict[str, Any]]:
    """Convert Realtime items to chat messages; vLLM applies the chat template.

    Function calls join the assistant message they follow, as one chat
    completion turn holds both the spoken text and its tool calls.
    """
    messages: list[dict[str, Any]] = []
    for item in items:
        if item["type"] == "function_call":
            previous = messages[-1] if messages else None
            if previous is None or previous["role"] != "assistant":
                previous = {"role": "assistant", "content": ""}
                messages.append(previous)
            previous.setdefault("tool_calls", []).append(_tool_call(item))
        elif item["type"] == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "content": item["output"],
                }
            )
        else:
            messages.append(
                {
                    "role": item["role"],
                    "content": "".join(part["text"] for part in item["content"]),
                }
            )
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})
    return messages


def _reasoning_options(value: Any) -> dict[str, Any]:
    """Map Realtime ``reasoning.effort`` to the chat template ``enable_thinking`` switch.

    Chat templates turn thinking on or off, and Realtime has no ``none`` effort,
    so ``minimal`` turns it off and every other effort turns it on.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("reasoning must be an object")
    effort = value.get("effort")
    if effort is None:
        return {}
    if effort not in REASONING_EFFORTS:
        raise ValueError(
            f"reasoning.effort must be one of {', '.join(REASONING_EFFORTS)}"
        )
    return {"enable_thinking": effort != "minimal"}


def _chat_options(
    session: dict[str, Any], response: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Map session options, overridden by ``response`` ones, to chat completion fields."""
    response = response or {}
    options: dict[str, Any] = {}
    if thinking := _reasoning_options(
        response.get("reasoning", session.get("reasoning"))
    ):
        options["chat_template_kwargs"] = thinking
    options.update(
        _chat_tool_options(
            {
                "tools": response.get("tools", session.get("tools")),
                "tool_choice": response.get("tool_choice", session.get("tool_choice")),
                "parallel_tool_calls": response.get(
                    "parallel_tool_calls", session.get("parallel_tool_calls")
                ),
            }
        )
    )
    return options


def _chat_tool_options(options: dict[str, Any]) -> dict[str, Any]:
    """Map Realtime tool options to chat completion tool fields."""
    tools = options.get("tools") or []
    tool_choice = options.get("tool_choice")
    if not tools:
        if tool_choice in (None, "auto", "none"):
            return {}
        raise ValueError("tool_choice requires tools")
    if isinstance(tool_choice, dict) and tool_choice["name"] not in {
        tool["name"] for tool in tools
    }:
        raise ValueError(
            f"tool_choice function {tool_choice['name']!r} is not in tools"
        )
    chat: dict[str, Any] = {
        "tools": [
            {
                "type": "function",
                "function": {
                    key: tool[key]
                    for key in ("name", "description", "parameters")
                    if key in tool
                },
            }
            for tool in tools
        ]
    }
    if isinstance(tool_choice, dict):
        chat["tool_choice"] = {
            "type": "function",
            "function": {"name": tool_choice["name"]},
        }
    elif tool_choice is not None:
        chat["tool_choice"] = tool_choice
    if options.get("parallel_tool_calls") is not None:
        chat["parallel_tool_calls"] = options["parallel_tool_calls"]
    return chat


def _realtime_usage(usage: dict[str, Any] | None) -> dict[str, int] | None:
    if usage is None:
        return None
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
    }
