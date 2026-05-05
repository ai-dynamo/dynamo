# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared HTTP client for the e2e parity wrappers.

`parse(impl, family, raw_text, tools, base_url)` POSTs to a real
`vllm serve` / `sglang.launch_server` `/v1/chat/completions`,
constrains the response to emit `raw_text` verbatim, and returns
the server's parsed `tool_calls` JSON wrapped in a `ParseResult`.

Per-impl differences are limited to where the regex constraint
goes in the request body:

- vLLM (post-0.20):  `body["structured_outputs"]["regex"] = …`
- SGLang:            `body["regex"] = …`

Both servers expose the same OpenAI-style chat-completions API
otherwise; this module dispatches on `impl` and reuses everything
else (request shape, response parsing, error handling, max-tokens
budget).
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from tests.parity.common import ParseResult
from tests.parity.parser.server import resolve_model

_REQUEST_TIMEOUT = 180.0


def parse(
    impl: str,
    parser_family: str,
    raw_text: str,
    tools: list[dict[str, Any]] | None,
    *,
    base_url: str,
) -> ParseResult:
    """POST a chat completion that's constrained to emit `raw_text`,
    then return whatever the server's parser extracted.
    """
    model = resolve_model(parser_family)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "force"}],
        "tools": _wrap_tools(tools),
        "tool_choice": "auto",
        "max_tokens": _max_tokens_for(raw_text),
        "temperature": 0.0,
    }
    _apply_regex_constraint(body, impl, re.escape(raw_text))

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        return ParseResult(error=f"{type(e).__name__}: {e}")
    if resp.status_code != 200:
        return ParseResult(error=f"HTTP {resp.status_code}: {resp.text[:500]}")

    msg = resp.json()["choices"][0]["message"]
    calls: list[dict[str, Any]] = []
    for tc in msg.get("tool_calls") or []:
        args_str = tc["function"]["arguments"]
        try:
            args = json.loads(args_str) if args_str else {}
        except (json.JSONDecodeError, TypeError):
            args = args_str
        calls.append({"name": tc["function"]["name"], "arguments": args})

    normal_text = msg.get("content") or ""
    return ParseResult(calls=calls, normal_text=normal_text)


def _apply_regex_constraint(body: dict[str, Any], impl: str, regex: str) -> None:
    """Set the constrained-decoding regex in the impl's expected location."""
    if impl == "vllm":
        body["structured_outputs"] = {"regex": regex}
    elif impl == "sglang":
        body["regex"] = regex
    else:
        raise ValueError(f"unknown impl: {impl!r}")


def _wrap_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Wrap flat tool defs in OpenAI Tool[] shape."""
    if not tools:
        return None
    wrapped = []
    for t in tools:
        if "function" in t:
            wrapped.append(t)
        else:
            wrapped.append({"type": "function", "function": t})
    return wrapped


def _max_tokens_for(text: str) -> int:
    """Conservative token budget for the constrained output.

    The regex forces exact bytes; the engine still needs `max_tokens`
    headroom to emit them. 3 chars/token average + 64-token slack
    handles all current fixtures.
    """
    return min(2048, max(64, len(text)) * 3 + 64)
