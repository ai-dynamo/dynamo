# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""parser-mode wrapper for Dynamo's Rust parser, via the PyO3 binding."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from dynamo._core import parse_tool_call
from tests.parser_parity.impls.common import ParseResult

# Maps parser_family (the fixture-corpus name) → the parser_name that
# Dynamo's PyO3 binding expects. Today these are 1:1; expand as the
# corpus grows.
_FAMILY_TO_DYNAMO_NAME = {
    "kimi_k2": "kimi_k2",
    "qwen3_coder": "qwen3_coder",
    "minimax_m2": "minimax_m2",
    "nemotron_deci": "nemotron_deci",
    "glm47": "glm47",
    "deepseek_v3_1": "deepseek_v3_1",
    "harmony": "harmony",
}


def parse(
    parser_family: str,
    raw_text: str,
    tools: list[dict[str, Any]] | None,
) -> ParseResult:
    parser_name = _FAMILY_TO_DYNAMO_NAME.get(parser_family, parser_family)
    tools_json = json.dumps(tools) if tools else None

    try:
        # The PyO3 binding returns a future that registers with a running
        # event loop, so we must call it from inside an async context.
        async def _run() -> str:
            return await parse_tool_call(parser_name, raw_text, tools_json)

        result_json: str = asyncio.run(_run())
    except Exception as e:
        return ParseResult(error=f"{type(e).__name__}: {e}")

    raw = json.loads(result_json)
    calls = []
    for c in raw.get("calls") or []:
        # Dynamo emits arguments as a JSON-encoded string; decode for canonical compare.
        fn = c.get("function", {})
        args_str = fn.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = args_str  # leave as-is — diff will surface the mismatch
        calls.append({"name": fn.get("name", ""), "arguments": args})

    return ParseResult(calls=calls, normal_text=raw.get("normal_text"))
