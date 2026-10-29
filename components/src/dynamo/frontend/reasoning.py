# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from jinja2.exceptions import TemplateError


def _requires_reasoning_string(
    render_template: Callable[..., str], has_tools: bool
) -> bool:
    # Match the Rust renderer's probe: array repr escapes these control bytes,
    # whereas a template that consumes the segments preserves them.
    first = "\x01dynamo_reasoning_probe_first\x01"
    last = "\x01dynamo_reasoning_probe_last\x01"
    tool = {"type": "function", "function": {"name": "probe", "parameters": {}}}

    def render(reasoning: str | list[str]) -> str:
        return render_template(
            [
                {"role": "user", "content": "u"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": reasoning,
                    "tool_calls": [
                        {
                            "id": "call_probe",
                            "type": "function",
                            "function": {"name": "probe", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_probe", "content": "r"},
            ],
            tools=[tool] if has_tools else None,
        )

    try:
        if first not in render(first):
            return False
    except (TemplateError, TypeError, ValueError):
        # A probe cannot supply every model-specific template input. Let the
        # real request render, including its normal validation errors.
        return False
    try:
        array_output = render([first, last])
    except (TemplateError, TypeError, ValueError):
        return True
    return first not in array_output and last not in array_output


def normalize_reasoning_content(
    messages: list[dict[str, Any]],
    *,
    render_template: Callable[..., str] | None = None,
    has_tools: bool = False,
    string_only: bool = False,
) -> None:
    """Adapt copied history using the real renderer's effective request options.

    Do not share capability results across requests with different render options.
    """
    segmented = [
        msg
        for msg in messages
        if msg.get("role") == "assistant"
        and isinstance(msg.get("reasoning_content"), list)
    ]
    if not segmented:
        return
    if not string_only:
        if render_template is None or not _requires_reasoning_string(
            render_template, has_tools
        ):
            return
    for msg in segmented:
        msg["reasoning_content"] = "\n".join(
            segment for segment in msg["reasoning_content"] if segment
        )
