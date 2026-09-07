#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Shared behavioral cases for backend tool-guidance tests."""

from dataclasses import dataclass
from typing import Any, Literal

GuidanceSource = Literal["none", "assistant", "tool"]
ToolChoice = Literal["auto", "none", "required", "named"]


@dataclass(frozen=True)
class ToolGuidanceParityCase:
    name: str
    tool_choice: ToolChoice
    has_tools: bool
    has_assistant_constraint: bool
    expected: GuidanceSource
    # Backends known to disagree with `expected` today, mapped to what they
    # actually produce. The row still states the correct answer; a listed backend
    # asserts against its recorded value instead, so the code under test still
    # runs and any OTHER change in its behavior fails loudly. Fixing a backend
    # means deleting its entry here, not editing `expected`.
    known_divergent: tuple[tuple[str, GuidanceSource], ...] = ()

    def divergent_source(self, backend: str) -> GuidanceSource | None:
        for name, source in self.known_divergent:
            if name == backend:
                return source
        return None


TOOL_GUIDANCE_PARITY_CASES = (
    ToolGuidanceParityCase("no-tools", "auto", False, False, "none"),
    ToolGuidanceParityCase("tool-choice-none", "none", True, False, "none"),
    ToolGuidanceParityCase("tool-choice-required", "required", True, False, "tool"),
    ToolGuidanceParityCase("named-tool-choice", "named", True, False, "tool"),
    ToolGuidanceParityCase("automatic-tool-choice", "auto", True, False, "tool"),
    ToolGuidanceParityCase(
        "response-format-precedence",
        "auto",
        True,
        True,
        "assistant",
    ),
    # response_format is scoped to the message returned to the user, not to tool
    # calls, so a forced choice keeps the tool constraint and drops it. SGLang
    # instead lets response_format win and can return no tool call at all --
    # see the TODO above `guided_decoding = response_format_guided_decoding or ...`
    # in sglang_prepost.py.
    ToolGuidanceParityCase(
        "forced-choice-with-response-format",
        "required",
        True,
        True,
        "tool",
        known_divergent=(("sglang", "assistant"),),
    ),
    ToolGuidanceParityCase(
        "named-choice-with-response-format",
        "named",
        True,
        True,
        "tool",
        known_divergent=(("sglang", "assistant"),),
    ),
)


def parity_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }


def tool_choice_value(tool_choice: ToolChoice) -> str | dict[str, Any]:
    if tool_choice == "named":
        return {
            "type": "function",
            "function": {"name": "get_weather"},
        }
    return tool_choice


def assistant_response_format() -> dict[str, str]:
    return {"type": "json_object"}


def classify_guidance_source(
    guided_decoding: dict[str, Any] | None,
    *,
    has_assistant_constraint: bool,
) -> GuidanceSource:
    if guided_decoding is None:
        return "none"
    if has_assistant_constraint and guided_decoding == {"json": {"type": "object"}}:
        return "assistant"
    return "tool"
