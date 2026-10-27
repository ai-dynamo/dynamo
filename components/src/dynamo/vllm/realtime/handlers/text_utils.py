# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Realtime text options and normalize conversation items and usage."""

from __future__ import annotations

import json
import uuid
from typing import Any


def _max_output_tokens(value: Any) -> tuple[int | None, int | str]:
    if value in (None, "inf"):
        return None, "inf"
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4096:
        raise ValueError("max_output_tokens must be an integer from 1 to 4096 or 'inf'")
    return value, value


def _validate_text_options(options: dict[str, Any], name: str) -> None:
    if options.get("output_modalities") not in (None, ["text"]):
        raise ValueError("only text output is supported")
    if options.get("prompt") is not None:
        raise ValueError("prompt templates are not supported")
    if not isinstance(options.get("instructions", ""), str):
        raise ValueError(f"{name}.instructions must be a string")


def _normalize_text_item(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("conversation item must be an object")
    item_type = value.get("type")
    if item_type in ("function_call", "function_call_output"):
        fields = (
            ("call_id", "name", "arguments")
            if item_type == "function_call"
            else ("call_id", "output")
        )
        if any(not isinstance(value.get(key), str) for key in fields):
            raise ValueError("function item fields must be strings")
        if not value["call_id"] or (item_type == "function_call" and not value["name"]):
            raise ValueError("function call identity must not be empty")
        if item_type == "function_call":
            if value.get("status", "completed") != "completed":
                raise ValueError("only completed function calls may be replayed")
            try:
                arguments = json.loads(value["arguments"])
            except json.JSONDecodeError as exc:
                raise ValueError("function arguments must be complete JSON") from exc
            if not isinstance(arguments, dict):
                raise ValueError("function arguments must be a JSON object")
        item_id = value.get("id") or f"item_{uuid.uuid4().hex}"
        if not isinstance(item_id, str):
            raise ValueError("conversation item id must be a string")
        return {
            "id": item_id,
            "object": "realtime.item",
            "type": item_type,
            "status": "completed",
            **{key: value[key] for key in fields},
        }
    if item_type != "message":
        raise ValueError("only text messages and function items are supported")
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

    item = {
        "id": value.get("id") or f"item_{uuid.uuid4().hex}",
        "object": "realtime.item",
        "type": "message",
        "status": "completed",
        "role": role,
        "content": [{"type": content_type, "text": part["text"]} for part in content],
    }
    if not isinstance(item["id"], str):
        raise ValueError("conversation item id must be a string")
    return item


def _text_prompt(
    items: list[dict[str, Any]], instructions: str, *, require_results: bool = False
) -> list[dict[str, Any]]:
    """Convert Realtime items to chat messages; vLLM applies the chat template."""
    messages: list[dict[str, Any]] = []
    calls: set[str] = set()
    results: set[str] = set()
    for item in items:
        if item["type"] == "function_call":
            if item["status"] != "completed":
                raise ValueError(
                    "incomplete function calls require conversation resynchronization"
                )
            call_id = item["call_id"]
            if call_id in calls:
                raise ValueError("duplicate function call_id")
            calls.add(call_id)
            if not messages or messages[-1]["role"] != "assistant":
                messages.append({"role": "assistant", "content": None})
            messages[-1].setdefault("tool_calls", []).append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                }
            )
        elif item["type"] == "function_call_output":
            call_id = item["call_id"]
            if call_id not in calls or call_id in results:
                raise ValueError("function output must match one unresolved call")
            results.add(call_id)
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": item["output"]}
            )
        else:
            content = "".join(part["text"] for part in item["content"])
            if (
                item["role"] == "assistant"
                and messages
                and messages[-1]["role"] == "assistant"
            ):
                messages[-1]["content"] = (messages[-1]["content"] or "") + content
            else:
                if calls != results:
                    raise ValueError(
                        "function outputs are required before another message"
                    )
                messages.append({"role": item["role"], "content": content})
    if require_results and calls != results:
        raise ValueError("function outputs are required before generation")
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})
    return messages


def _realtime_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    result: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
    }
    details = usage.get("prompt_tokens_details") or {}
    if details.get("cached_tokens") is not None:
        result["input_token_details"] = {"cached_tokens": details["cached_tokens"]}
    return result
