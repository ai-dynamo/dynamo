# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize tool policies from the payload actually sent to an endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolPolicy:
    mode: str
    allowed_names: frozenset[str] | None
    max_calls: int | None
    requires_call: bool
    named_name: str | None = None
    allowed_identities: frozenset[tuple[str | None, str]] | None = None

    def allows(self, name: str, namespace: str | None = None) -> bool:
        if self.allowed_identities is not None:
            return (namespace, name) in self.allowed_identities
        return self.allowed_names is None or name in self.allowed_names


def _function_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function", tool)
    if not isinstance(function, Mapping):
        raise ValueError("function selection must be an object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("function selection must have a nonempty name")
    return name


def _function_identity(tool: Mapping[str, Any]) -> tuple[str | None, str]:
    name = _function_name(tool)
    function = tool.get("function", tool)
    namespace = function.get("namespace", tool.get("namespace"))
    if namespace is not None and (not isinstance(namespace, str) or not namespace):
        raise ValueError("function namespace must be a nonempty string or null")
    return namespace, name


def effective_tool_policy(payload: Mapping[str, Any]) -> ToolPolicy:
    """Accept Chat's nested and Responses' flat function-choice shapes."""

    choice = payload.get("tool_choice")
    mode = "auto"
    allowed_names = None
    allowed_identities = None
    named_name = None
    if isinstance(choice, str):
        if choice not in {"auto", "required", "none"}:
            raise ValueError(f"unsupported tool_choice {choice!r}")
        mode = choice
    elif isinstance(choice, Mapping):
        choice_type = choice.get("type")
        if choice_type == "function":
            mode = "named"
            identity = _function_identity(choice)
            named_name = identity[1]
            allowed_names = frozenset({named_name})
            allowed_identities = frozenset({identity})
        elif choice_type == "allowed_tools":
            selection = choice.get("allowed_tools", choice)
            if not isinstance(selection, Mapping):
                raise ValueError("allowed_tools selection must be an object")
            mode = selection.get("mode")
            if not isinstance(mode, str) or mode not in {"auto", "required"}:
                raise ValueError("allowed_tools mode must be auto or required")
            tools = selection.get("tools")
            if not isinstance(tools, list) or not all(
                isinstance(tool, Mapping) and tool.get("type") == "function"
                for tool in tools
            ):
                raise ValueError("allowed_tools must select function tools")
            allowed_identities = frozenset(_function_identity(tool) for tool in tools)
            allowed_names = frozenset(name for _, name in allowed_identities)
        else:
            raise ValueError(f"unsupported tool_choice type {choice_type!r}")
    elif choice is not None:
        raise ValueError("tool_choice must be a string, object or null")
    if mode == "none":
        allowed_names = frozenset()
        allowed_identities = frozenset()
    return ToolPolicy(
        mode=mode,
        allowed_names=allowed_names,
        max_calls=1 if payload.get("parallel_tool_calls") is False else None,
        requires_call=mode in {"required", "named"},
        named_name=named_name,
        allowed_identities=allowed_identities,
    )
