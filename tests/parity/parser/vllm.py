# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""parser-mode wrapper for vLLM's Python tool parsers (in-process import)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

# vLLM's tool-parser module path settled on `vllm.tool_parsers.*` in v0.6.x;
# the older `vllm.entrypoints.openai.tool_parsers.*` location was retained as
# an alias for a few releases. Try the new location first.
try:
    from vllm.tool_parsers import (  # type: ignore[import-untyped]
        ToolParser,
        ToolParserManager,
    )
except ImportError:
    from vllm.entrypoints.openai.tool_parsers import (  # type: ignore[import-untyped]
        ToolParser,
        ToolParserManager,
    )

from tests.parity.common import ParseResult, decode_arguments


class _OmnivorousVocab(dict):
    """Returns a non-None synthetic ID for any token lookup.

    Several vLLM parsers (qwen3_coder, glm47_moe, minimax_m2, deepseekv31,
    etc.) call `self.vocab.get(token_string)` at __init__ and refuse to
    construct if the result is `None`. Real tokenizers return real IDs,
    but `extract_tool_calls()` (text mode) only uses the IDs to compare
    against `delta_token_ids`, which we never pass — so any non-None int
    works.
    """

    def get(self, key, default=None):  # type: ignore[override]
        return 1


_STUB_VOCAB = _OmnivorousVocab()


class _StubTokenizer:
    """Truthy placeholder for vLLM parsers that do text-based extraction.

    Most vLLM `extract_tool_calls()` implementations don't actually need a
    tokenizer, but the parser __init__s look up special-token IDs via
    `self.vocab.get(...)`. We return synthetic non-None IDs to satisfy
    those checks; the IDs aren't used during text extraction.
    """

    def get_vocab(self) -> dict[str, int]:
        return _STUB_VOCAB


# Maps parser_family → vLLM's registered parser key (registered via
# @ToolParserManager.register_module(<key>)).
_FAMILY_TO_VLLM_KEY = {
    "kimi_k2": "kimi_k2",
    "qwen3_coder": "qwen3_coder",
    "minimax_m2": "minimax_m2",
    "glm47": "glm47",
    "deepseek_v3_1": "deepseek_v31",
    "harmony": "openai",  # gpt-oss / harmony parser registered as "openai"
}


def parse(
    parser_family: str,
    raw_text: str,
    tools: list[dict[str, Any]] | None,
) -> ParseResult:
    key = _FAMILY_TO_VLLM_KEY.get(parser_family)
    if key is None:
        return ParseResult(error=f"vLLM has no parser for family={parser_family!r}")

    parser_cls = ToolParserManager.get_tool_parser(key)

    # vLLM's ToolParser constructor checks `if not self.model_tokenizer:` and raises
    # if falsy. None of the parsers we test actually call tokenizer methods inside
    # extract_tool_calls(), so a truthy stub satisfies the check.
    parser: ToolParser = parser_cls(tokenizer=_StubTokenizer())

    # vLLM's extract_tool_calls signature: (model_output, request) → ExtractedToolCallInformation.
    # We construct a minimal request shape with only the tools field, since most parsers ignore the rest.
    request = _build_chat_request(tools)

    try:
        info = parser.extract_tool_calls(raw_text, request)
    except Exception as e:
        return ParseResult(error=f"{type(e).__name__}: {e}")

    calls = [
        {"name": tc.function.name, "arguments": decode_arguments(tc.function.arguments)}
        for tc in info.tool_calls or []
    ]
    return ParseResult(calls=calls, normal_text=info.content)


def _build_chat_request(tools: list[dict[str, Any]] | None) -> Any:
    """Minimal duck-typed request shape — vLLM's parsers only read `.tools`."""
    if not tools:
        return SimpleNamespace(tools=None)
    wrapped = [
        t if "function" in t else {"type": "function", "function": t} for t in tools
    ]
    return SimpleNamespace(tools=wrapped)
