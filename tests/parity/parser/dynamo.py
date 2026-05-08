# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""parser-mode wrapper for Dynamo's Rust parser, via the PyO3 binding."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from dynamo._core import parse_tool_call
from tests.parity.common import ParseResult, decode_arguments


def parse(
    parser_family: str,
    raw_text: str,
    tools: list[dict[str, Any]] | None,
) -> ParseResult:
    tools_json = json.dumps(tools) if tools else None

    try:
        # The PyO3 binding returns a future that registers with a running
        # event loop, so we must call it from inside an async context.
        async def _run() -> str:
            return await parse_tool_call(parser_family, raw_text, tools_json)

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
