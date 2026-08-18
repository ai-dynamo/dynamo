# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.input_params module."""

from typing import Any

import pytest

from dynamo.common.utils.input_params import InputParamManager

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class RecordingTokenizer:
    """Captures the keyword arguments the manager renders the template with."""

    chat_template = ""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        self.kwargs = kwargs
        return "rendered"


TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]

MESSAGES = [{"role": "user", "content": "What is the weather in Tokyo?"}]


def test_tools_are_forwarded_to_the_chat_template():
    # Without this the worker templates as if no tools existed, so the model
    # cannot emit a tool call and the request returns finish_reason=stop.
    tokenizer = RecordingTokenizer()

    InputParamManager(tokenizer).get_input_param(
        {"messages": MESSAGES, "tools": TOOLS}, use_tokenizer=True
    )

    assert tokenizer.kwargs.get("tools") == TOOLS


def test_absent_tools_are_not_passed_to_the_chat_template():
    # A request without tools must render exactly as before.
    tokenizer = RecordingTokenizer()

    InputParamManager(tokenizer).get_input_param(
        {"messages": MESSAGES}, use_tokenizer=True
    )

    assert "tools" not in tokenizer.kwargs


def test_tool_choice_none_keeps_tools_out_of_the_template():
    # The ModelInput::Tokens path drops tool_dicts for tool_choice="none" so the
    # model does not see the tools and emit raw XML tool calls in its prose.
    # Rendering them here would make the two paths disagree about "none".
    tokenizer = RecordingTokenizer()

    InputParamManager(tokenizer).get_input_param(
        {"messages": MESSAGES, "tools": TOOLS, "tool_choice": "none"},
        use_tokenizer=True,
    )

    assert "tools" not in tokenizer.kwargs


@pytest.mark.parametrize("tool_choice", ["auto", "required"])
def test_tools_are_forwarded_for_non_none_tool_choice(tool_choice):
    tokenizer = RecordingTokenizer()

    InputParamManager(tokenizer).get_input_param(
        {"messages": MESSAGES, "tools": TOOLS, "tool_choice": tool_choice},
        use_tokenizer=True,
    )

    assert tokenizer.kwargs.get("tools") == TOOLS


def test_explicit_chat_template_kwargs_tools_win():
    # chat_template_kwargs is a deliberate escape hatch, and forwarding the
    # request's tools as a keyword alongside it would raise
    # "got multiple values for keyword argument 'tools'".
    tokenizer = RecordingTokenizer()
    override = [{"type": "function", "function": {"name": "override"}}]

    InputParamManager(tokenizer).get_input_param(
        {
            "messages": MESSAGES,
            "tools": TOOLS,
            "chat_template_kwargs": {"tools": override},
        },
        use_tokenizer=True,
    )

    assert tokenizer.kwargs.get("tools") == override
