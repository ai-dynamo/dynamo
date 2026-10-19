# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Realtime text options and normalize conversation items and usage."""

from __future__ import annotations

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
    if options.get("tools") not in (None, []):
        raise ValueError("tools are not supported")
    if options.get("tool_choice") not in (None, "none"):
        raise ValueError("tool_choice is not supported")
    if any(options.get(field) is not None for field in ("prompt", "reasoning")):
        raise ValueError("prompt and reasoning configuration are not supported")
    if not isinstance(options.get("instructions", ""), str):
        raise ValueError(f"{name}.instructions must be a string")


def _normalize_text_item(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "message":
        raise ValueError("only message conversation items are supported")
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
    items: list[dict[str, Any]], instructions: str
) -> list[dict[str, str]]:
    """Convert Realtime items to chat messages; vLLM applies the chat template."""
    messages = [
        {
            "role": item["role"],
            "content": "".join(part["text"] for part in item["content"]),
        }
        for item in items
    ]
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})
    return messages


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
