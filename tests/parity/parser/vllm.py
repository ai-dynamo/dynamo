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

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam

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
    "deepseek_v3": "deepseek_v3",
    "deepseek_v3_2": "deepseek_v32",
    "deepseek_v4": "deepseek_v4",
    "hermes": "hermes",
    "mistral": "mistral",
    "jamba": "jamba",
    "llama3_json": "llama3_json",
    "phi4": "phi4_mini_json",
    "harmony": "openai",  # gpt-oss / harmony parser registered as "openai"
    "pythonic": "pythonic",
    "gemma4": "gemma4",
}


def _tools_as_namespace(
    tools: list[dict[str, Any]] | None,
) -> list[SimpleNamespace] | None:
    """Wrap tool dicts so vLLM parsers can use attribute access
    (``tool.function.name``) while keeping ``parameters`` as a plain
    dict (vLLM also calls ``.get()`` on it)."""
    if not tools:
        return None
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append(
            SimpleNamespace(
                type=t.get("type", "function"),
                function=SimpleNamespace(
                    name=fn["name"],
                    parameters=fn.get("parameters") or {},
                ),
            )
        )
    return out


def parse(
    parser_family: str,
    raw_text: str,
    tools: list[dict[str, Any]] | None,
) -> ParseResult:
    key = _FAMILY_TO_VLLM_KEY.get(parser_family)
    if key is None:
        return ParseResult(
            error=f"UNAVAILABLE: vLLM has no parser for family={parser_family!r}"
        )

    # Wrap flat tool defs as real `ChatCompletionToolsParam` Pydantic instances
    wrapped_tools = (
        [
            ChatCompletionToolsParam.model_validate(
                t if "function" in t else {"type": "function", "function": t}
            )
            for t in tools
        ]
        if tools
        else None
    )
    # Some vLLM parsers (e.g. glm4_moe) access request.tools[i].function.name
    # via attribute, not dict subscript. Convert to SimpleNamespace so both
    # access styles work.
    ns_tools = _tools_as_namespace(wrapped_tools)
    try:
        parser_cls = ToolParserManager.get_tool_parser(key)
        # vLLM's ToolParser constructor checks `if not self.model_tokenizer:` and raises
        # if falsy. None of the parsers we test actually call tokenizer methods inside
        # extract_tool_calls(), so a truthy stub satisfies the check.
        parser: ToolParser = parser_cls(tokenizer=_StubTokenizer(), tools=ns_tools)
        # vLLM's extract_tool_calls signature: (model_output, request) → ExtractedToolCallInformation.
        # We construct a minimal request shape with only the tools field, since most parsers ignore the rest.
        request = SimpleNamespace(tools=ns_tools)
        info = parser.extract_tool_calls(raw_text, request)
    except NotImplementedError as e:
        # Known unsupported combinations (e.g., vLLM's harmony parser requires
        # token IDs, not text). Treat as env-unavailable so the harness skips
        # rather than failing as a regression.
        return ParseResult(error=f"UNAVAILABLE: {type(e).__name__}: {e}")
    except Exception as e:
        return ParseResult(error=f"{type(e).__name__}: {e}")

    calls = [
        {"name": tc.function.name, "arguments": decode_arguments(tc.function.arguments)}
        for tc in info.tool_calls or []
    ]
    return ParseResult(calls=calls, normal_text=info.content)
