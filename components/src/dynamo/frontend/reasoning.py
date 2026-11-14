# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections.abc import Callable
from typing import Any

from jinja2.exceptions import TemplateError


def _requires_reasoning_string(
    render_template: Callable[..., str], tools: list[dict[str, Any]] | None
) -> bool:
    # Match the Rust renderer: Jinja interpolation and ~ stringify lists with
    # literal "\\x01", not byte 0x01. Compare complete sentinels including the
    # control bytes, not just their printable names.
    first = "\x01dynamo_reasoning_probe_first\x01"
    last = "\x01dynamo_reasoning_probe_last\x01"

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
            tools=tools,
        )

    try:
        scalar_output = render(first)
        json_first = json.dumps(first)[1:-1]
        if first not in scalar_output and json_first not in scalar_output:
            return False
    except (TemplateError, TypeError, ValueError):
        # A probe cannot supply every model-specific template input. Let the
        # real request render, including its normal validation errors.
        return False
    try:
        array_output = render([first, last])
    except (TemplateError, TypeError, ValueError):
        return True
    if first in array_output or last in array_output:
        return False
    json_last = json.dumps(last)[1:-1]
    if json_first in array_output or json_last in array_output:
        # A JSON array is container serialization; separately serialized
        # segments still allow a template to interleave reasoning and tools.
        json_array = (
            rf'\[\s*"{re.escape(json_first)}"\s*,\s*"{re.escape(json_last)}"\s*\]'
        )
        return bool(re.search(json_array, array_output))
    return True


def normalize_reasoning_content(
    messages: list[dict[str, Any]],
    *,
    render_template: Callable[..., str] | None = None,
    tools: list[dict[str, Any]] | None = None,
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
            render_template, tools
        ):
            return
    for msg in segmented:
        msg["reasoning_content"] = "\n".join(
            segment for segment in msg["reasoning_content"] if segment
        )
