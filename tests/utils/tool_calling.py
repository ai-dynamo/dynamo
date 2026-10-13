# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tool-call response checks for frontend and native-sidecar tests."""

import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class StreamResult:
    content: str
    reasoning_content: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None
    model: str
    chunks: int
    ttft_ms: float
    raw_chunks: list[Any]


def collect_stream(stream) -> StreamResult:
    """Reconstruct one chat choice without hiding invalid tool metadata."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    argument_parts: dict[int, list[str]] = {}
    finish_reason = None
    model = ""
    chunk_count = 0
    raw_chunks: list[Any] = []
    t0 = time.monotonic()
    ttft_ms = 0.0

    for chunk in stream:
        raw_chunks.append(chunk)
        chunk_count += 1
        if chunk_count == 1:
            ttft_ms = (time.monotonic() - t0) * 1000.0
        model = chunk.model

        for choice in chunk.choices:
            delta = choice.delta

            if getattr(delta, "content", None):
                content_parts.append(delta.content)

            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)

            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index
                    entry = tool_calls_by_index.setdefault(
                        idx,
                        {
                            "id": "",
                            "type": None,
                            "function": {"name": "", "arguments": ""},
                        },
                    )

                    if tc.id:
                        if entry["id"] and entry["id"] != tc.id:
                            raise AssertionError(
                                f"Tool call id changed within same index {idx}: "
                                f"{entry['id']} -> {tc.id}"
                            )
                        entry["id"] = tc.id

                    if tc.type:
                        assert entry["type"] in (None, tc.type), "Tool type changed"
                        entry["type"] = tc.type

                    if tc.function:
                        if tc.function.name:
                            if (
                                entry["function"]["name"]
                                and entry["function"]["name"] != tc.function.name
                            ):
                                raise AssertionError(
                                    f"Tool name changed within same index {idx}: "
                                    f"{entry['function']['name']} -> {tc.function.name}"
                                )
                            entry["function"]["name"] = tc.function.name

                        if tc.function.arguments:
                            argument_parts.setdefault(idx, []).append(
                                tc.function.arguments
                            )

            if choice.finish_reason:
                assert finish_reason is None, "Multiple terminal finish reasons"
                finish_reason = choice.finish_reason

    for idx, entry in tool_calls_by_index.items():
        entry["function"]["arguments"] = "".join(argument_parts.get(idx, []))
    ordered_tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    return StreamResult(
        content="".join(content_parts),
        reasoning_content="".join(reasoning_parts),
        tool_calls=ordered_tool_calls,
        finish_reason=finish_reason,
        model=model,
        chunks=chunk_count,
        ttft_ms=ttft_ms,
        raw_chunks=raw_chunks,
    )


def parse_and_validate_tool_call(
    tc: dict[str, Any],
    schema_by_name: dict[str, dict[str, Any]],
    *,
    expected_name: str | None = None,
) -> dict[str, Any]:
    """Validate the emitted call against the request's function schema."""
    from jsonschema import Draft7Validator

    assert tc["type"] == "function", f"unexpected tool type: {tc['type']!r}"
    assert tc["id"], "tool call id must be non-empty"
    fn_name = tc["function"]["name"]
    assert fn_name, "tool call function name must be non-empty"

    if expected_name is not None:
        assert fn_name == expected_name, f"expected {expected_name!r}, got {fn_name!r}"

    assert fn_name in schema_by_name, f"unknown tool name {fn_name!r}"
    args_str = tc["function"]["arguments"]
    assert args_str, "tool call arguments must be non-empty"

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError as e:
        raise AssertionError(f"arguments are not valid JSON: {args_str!r}") from e

    assert isinstance(args, dict), f"arguments must decode to object, got {type(args)}"

    validator = Draft7Validator(schema_by_name[fn_name])
    errors = sorted(validator.iter_errors(args), key=lambda e: list(e.path))
    if errors:
        rendered = "; ".join(
            f"path={list(err.path)} message={err.message}" for err in errors
        )
        raise AssertionError(f"arguments failed schema validation: {rendered}")

    return args
