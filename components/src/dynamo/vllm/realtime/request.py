# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated chat requests shared by realtime generation and prefix warming."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TextGenerationRequest:
    """A turn's private snapshot; neither session updates nor ASR mutate it."""

    messages: list[dict[str, Any]]
    max_output_tokens: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


def chat_options(options: dict[str, Any]) -> dict[str, Any]:
    """Translate the admitted realtime text profile into vLLM chat options.

    The worker validates types and combinations before starting speculative work.
    vLLM remains responsible for model-specific constraints and rendering.
    """
    result: dict[str, Any] = {}
    tools = options.get("tools")
    names: set[str] = set()
    if tools is not None:
        if not isinstance(tools, list):
            raise ValueError("tools must be an array")
        converted = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise ValueError("only function tools are supported")
            if tool.keys() - {"type", "name", "description", "parameters"}:
                raise ValueError("unsupported function tool field")
            name = tool.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("function names must be non-empty and unique")
            if not isinstance(tool.get("description"), str):
                raise ValueError("function description must be a string")
            if not isinstance(tool.get("parameters"), dict):
                raise ValueError("function parameters must be a JSON Schema object")
            names.add(name)
            converted.append(
                {
                    "type": "function",
                    "function": {k: v for k, v in tool.items() if k != "type"},
                }
            )
        # vLLM rejects an empty tools array. Realtime uses it to clear tools.
        if converted:
            result["tools"] = converted

    choice = options.get("tool_choice")
    if choice is None:
        choice = "auto" if names else "none"
    if isinstance(choice, dict):
        if choice.keys() != {"type", "name"} or choice.get("type") != "function":
            raise ValueError("tool_choice must select a named function")
        if not isinstance(choice.get("name"), str) or choice["name"] not in names:
            raise ValueError("tool_choice names an unavailable function")
        result["tool_choice"] = {
            "type": "function",
            "function": {"name": choice["name"]},
        }
    elif choice in ("auto", "required", "none"):
        if choice != "none" and not names:
            raise ValueError("tool_choice requires tools")
        result["tool_choice"] = choice
    else:
        raise ValueError("unsupported tool_choice")

    parallel = options.get("parallel_tool_calls")
    if parallel is not None:
        if not isinstance(parallel, bool):
            raise ValueError("parallel_tool_calls must be a boolean")
        result["parallel_tool_calls"] = parallel

    reasoning = options.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict) or reasoning.keys() - {"effort"}:
            raise ValueError("reasoning supports only effort")
        effort = reasoning.get("effort")
        if effort not in ("minimal", "low", "medium", "high", "xhigh"):
            raise ValueError("unsupported reasoning effort")
        result["reasoning_effort"] = effort

    nvext = options.get("nvext", {})
    if nvext is None:
        nvext = {}
    if not isinstance(nvext, dict) or nvext.keys() - {
        "sampling",
        "response_format",
        "chat_template_kwargs",
        "required_capabilities",
    }:
        raise ValueError("unsupported realtime nvext option")
    required = nvext.get("required_capabilities", [])
    if not isinstance(required, list) or any(not isinstance(x, str) for x in required):
        raise ValueError("required_capabilities must be an array of strings")
    sampling = nvext.get("sampling", {})
    if sampling is None:
        sampling = {}
    if not isinstance(sampling, dict):
        raise ValueError("nvext.sampling must be an object")
    ranges = {
        "temperature": (0, 2),
        "top_p": (0, 1),
        "frequency_penalty": (-2, 2),
        "presence_penalty": (-2, 2),
        "repetition_penalty": (0, None),
    }
    for name, value in sampling.items():
        if name in ("seed", "top_k"):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if name == "top_k" and value != -1 and value < 1:
                raise ValueError("top_k must be -1 or positive")
            if name == "seed" and not -(2**63) <= value < 2**63:
                raise ValueError("seed must fit a signed 64-bit integer")
        elif name in ranges:
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
            lower, upper = ranges[name]
            if value < lower or (upper is not None and value > upper):
                raise ValueError(f"{name} is outside its supported range")
            if name in ("top_p", "repetition_penalty") and value == 0:
                raise ValueError(f"{name} must be positive")
        else:
            raise ValueError(f"unsupported sampling option: {name}")
        result[name] = value

    template = nvext.get("chat_template_kwargs")
    if template is not None:
        if not isinstance(template, dict) or template.keys() - {"enable_thinking"}:
            raise ValueError("only enable_thinking chat_template_kwargs is supported")
        if "enable_thinking" in template and not isinstance(
            template["enable_thinking"], bool
        ):
            raise ValueError("enable_thinking must be a boolean")
        result["chat_template_kwargs"] = template

    response_format = nvext.get("response_format")
    if response_format is not None:
        if not isinstance(response_format, dict):
            raise ValueError("response_format must be an object")
        kind = response_format.get("type")
        if kind not in ("text", "json_object", "json_schema"):
            raise ValueError("unsupported response_format")
        if response_format.keys() - (
            {"type", "json_schema"} if kind == "json_schema" else {"type"}
        ):
            raise ValueError("unsupported response_format field")
        if kind == "json_schema":
            schema = response_format.get("json_schema")
            if not isinstance(schema, dict) or not isinstance(
                schema.get("schema"), dict
            ):
                raise ValueError("json_schema requires a schema object")
            if schema.keys() - {"name", "description", "strict", "schema"}:
                raise ValueError("unsupported json_schema field")
            if not isinstance(schema.get("name"), str) or not schema["name"]:
                raise ValueError("json_schema requires a non-empty name")
            if "strict" in schema and not isinstance(schema["strict"], bool):
                raise ValueError("json_schema.strict must be a boolean")
            if "description" in schema and not isinstance(schema["description"], str):
                raise ValueError("json_schema.description must be a string")
        if kind != "text" and choice != "none":
            raise ValueError("structured answers require tool_choice='none'")
        result["response_format"] = response_format
    return deepcopy(result)


def merge_options(session: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Apply response overrides without mutating the session or its nested maps."""
    merged = deepcopy({**session, **response})
    if "nvext" not in response or response["nvext"] is None:
        return merged
    override = response["nvext"]
    if not isinstance(override, dict):
        raise ValueError("nvext must be an object")
    previous = session.get("nvext") or {}
    merged["nvext"] = {**deepcopy(previous), **deepcopy(override)}
    if "sampling" in override and override["sampling"] is not None:
        if not isinstance(override["sampling"], dict):
            raise ValueError("nvext.sampling must be an object")
        merged["nvext"]["sampling"] = {
            **(previous.get("sampling") or {}),
            **override["sampling"],
        }
    return merged
