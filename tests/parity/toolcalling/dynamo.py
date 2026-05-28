# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""parser-mode wrapper for Dynamo's Rust parser, via the PyO3 binding."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
from typing import Any

from tests.parity.common import ParseResult, decode_arguments

_DYNAMO_CORE_SPEC = (
    importlib.util.find_spec("dynamo._core")
    if importlib.util.find_spec("dynamo") is not None
    else None
)


def _load_dynamo_core() -> Any | None:
    if _DYNAMO_CORE_SPEC is None:
        return None
    try:
        return importlib.import_module("dynamo._core")
    except (ImportError, OSError):
        return None


_DYNAMO_CORE = _load_dynamo_core()


def parse_tool_calls_batch(
    parser_family: str,
    raw_text: str,
    tools: list[dict[str, Any]] | None,
) -> ParseResult:
    if _DYNAMO_CORE is None:
        return ParseResult(error="UNAVAILABLE: dynamo._core is not installed")

    tools_json = json.dumps(tools) if tools else None

    try:
        # The PyO3 binding returns a future that registers with a running
        # event loop, so we must call it from inside an async context.
        async def _run() -> str:
            return await _DYNAMO_CORE.parse_tool_calls_batch(
                parser_family, raw_text, tools_json
            )

        result_json: str = asyncio.run(_run())
        raw = json.loads(result_json)
    except Exception as e:
        return ParseResult(error=f"{type(e).__name__}: {e}")

    calls = [
        {
            "name": c["function"]["name"],
            "arguments": decode_arguments(c["function"]["arguments"]),
        }
        for c in raw.get("calls") or []
    ]
    return ParseResult(calls=calls, normal_text=raw.get("normal_text"))


def parse_tool_calls_stream(
    parser_family: str,
    chunks: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> ParseResult:
    if _DYNAMO_CORE is None:
        return ParseResult(error="UNAVAILABLE: dynamo._core is not installed")

    tools_json = json.dumps(tools) if tools else None

    try:

        async def _run() -> str:
            return await _DYNAMO_CORE.parse_tool_calls_stream(
                parser_family, json.dumps(chunks), tools_json
            )

        result_json: str = asyncio.run(_run())
        raw = json.loads(result_json)
    except Exception as e:
        return ParseResult(error=f"{type(e).__name__}: {e}")

    calls = [
        {
            "name": c["name"],
            "arguments": decode_arguments(c["arguments"]),
        }
        for c in raw.get("calls") or []
    ]
    return ParseResult(calls=calls, normal_text=raw.get("normal_text"))
