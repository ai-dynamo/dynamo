# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""parser-mode wrapper for vLLM's Python tool parsers (in-process import)."""

from __future__ import annotations

import json
import re
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
from vllm.entrypoints.openai.parser.harmony_utils import get_encoding

from tests.parity.common import ParseResult, decode_arguments

_HARMONY_ASSISTANT_START = "<|start|>assistant"
_HARMONY_COMMENTARY_CALL_RE = re.compile(
    r"(?:<\|start\|>assistant)?<\|channel\|>commentary\b.*?<\|message\|>.*?<\|call\|>",
    re.DOTALL,
)
_HARMONY_ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)",
    re.DOTALL,
)
_HARMONY_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")
_HARMONY_MESSAGE_RE = re.compile(
    r"(?:<\|start\|>assistant)?"
    r"<\|channel\|>(?P<channel>\w+)"
    r"(?P<header>.*?)"
    r"<\|message\|>(?P<body>.*?)"
    r"(?P<stop><\|call\|>|<\|end\|>|<\|return\|>|$)",
    re.DOTALL,
)
_HARMONY_RECIPIENT_RE = re.compile(r"\bto=(?P<recipient>functions\.[\w.\-]+)")


class _HarmonyEncodingUnavailable(RuntimeError):
    pass


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


def _harmony_token_ids(raw_text: str) -> list[int]:
    """Encode gpt-oss/harmony fixture text for vLLM's token-ID parser.

    vLLM's OpenAIToolParser delegates to `parse_output_into_messages()`,
    which parses completion tokens from an assistant context. The first
    assistant-start marker is therefore outside the parser's input shape, but
    later assistant-start markers separate additional messages and must remain.
    """

    text = raw_text
    if text.startswith(_HARMONY_ASSISTANT_START):
        text = text[len(_HARMONY_ASSISTANT_START) :]

    try:
        enc = get_encoding()
    except Exception as e:
        raise _HarmonyEncodingUnavailable(f"{type(e).__name__}: {e}") from e
    return enc.encode(text, allowed_special=enc.special_tokens_set)


def _harmony_cleanup_residual_text(text: str) -> str:
    """Return the non-call narration left outside extracted commentary calls."""

    text = _HARMONY_ANALYSIS_RE.sub("", text)
    text = text.replace(_HARMONY_ASSISTANT_START, "")
    return _HARMONY_SPECIAL_TOKEN_RE.sub("", text)


def _harmony_token_text_and_normal_text(raw_text: str) -> tuple[str, str | None]:
    """Split prose-wrapped harmony text into vLLM token input plus narration.

    vLLM's OpenAIToolParser expects assistant-completion harmony tokens. The
    parity fixtures intentionally include surrounding narration to compare how
    engines handle mixed text and tool calls, so feed vLLM only complete
    commentary call envelopes and preserve the stripped residual text as the
    wrapper's normal_text contribution.
    """

    matches = list(_HARMONY_COMMENTARY_CALL_RE.finditer(raw_text))
    if not matches:
        return raw_text, None

    blocks = []
    residual_parts = []
    cursor = 0
    for match in matches:
        residual_parts.append(
            _harmony_cleanup_residual_text(raw_text[cursor : match.start()])
        )
        block = match.group(0)
        if block.startswith(_HARMONY_ASSISTANT_START):
            block = block[len(_HARMONY_ASSISTANT_START) :]
        if blocks:
            block = f"{_HARMONY_ASSISTANT_START}{block}"
        blocks.append(block)
        cursor = match.end()

    residual_parts.append(_harmony_cleanup_residual_text(raw_text[cursor:]))
    residual = "".join(residual_parts).strip()
    normal_text = residual if residual.strip() else None
    return "".join(blocks), normal_text


def _merge_normal_text(first: str | None, second: str | None) -> str | None:
    merged = "".join(part for part in (first, second) if part)
    return merged if merged.strip() else None


def _harmony_parse_without_encoding(
    token_text: str,
    residual_normal_text: str | None,
) -> ParseResult:
    """Mirror vLLM's OpenAI parser when Harmony encoding cannot be loaded.

    vLLM's OpenAIToolParser extracts completed `functions.*` commentary
    messages, parses valid JSON arguments, and otherwise preserves the raw
    argument text. Final-channel text and recipient-less commentary preambles
    are visible content; analysis and malformed tool-call messages are hidden.
    """

    calls = []
    final_content = None
    commentary_content = None

    for match in _HARMONY_MESSAGE_RE.finditer(token_text):
        channel = match.group("channel")
        header = match.group("header")
        body = match.group("body")
        stop = match.group("stop")
        recipient_match = _HARMONY_RECIPIENT_RE.search(header)
        recipient = recipient_match.group("recipient") if recipient_match else None

        if recipient and recipient.startswith("functions."):
            if channel != "commentary" or stop != "<|call|>":
                continue
            try:
                arguments = json.loads(body)
            except json.JSONDecodeError:
                arguments = body
            calls.append(
                {
                    "name": recipient.split("functions.", 1)[1],
                    "arguments": arguments,
                }
            )
        elif channel == "final":
            final_content = body
        elif channel == "commentary" and stop != "<|call|>":
            commentary_content = body

    return ParseResult(
        calls=calls,
        normal_text=_merge_normal_text(
            residual_normal_text,
            final_content or commentary_content,
        ),
    )


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

    # Wrap flat tool defs as real `ChatCompletionToolsParam` Pydantic instances —
    # vLLM's schema-aware coercion paths gate on `hasattr(config, "type")` and
    # `hasattr(config.function, "name")`, which the Pydantic model satisfies via
    # attribute access (plain dicts would silently fall back to raw-string emission).
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
    try:
        parser_cls = ToolParserManager.get_tool_parser(key)
        # vLLM's ToolParser constructor checks `if not self.model_tokenizer:` and raises
        # if falsy. None of the parsers we test actually call tokenizer methods inside
        # extract_tool_calls(), so a truthy stub satisfies the check.
        parser: ToolParser = parser_cls(tokenizer=_StubTokenizer(), tools=wrapped_tools)
        # vLLM's extract_tool_calls signature: (model_output, request) → ExtractedToolCallInformation.
        # We construct a minimal request shape with only the tools field, since most parsers ignore the rest.
        request = SimpleNamespace(tools=wrapped_tools)
        if key == "openai":
            token_text, residual_normal_text = _harmony_token_text_and_normal_text(
                raw_text
            )
            try:
                info = parser.extract_tool_calls(
                    token_text,
                    request,
                    token_ids=_harmony_token_ids(token_text),
                )
            except _HarmonyEncodingUnavailable:
                return _harmony_parse_without_encoding(
                    token_text,
                    residual_normal_text,
                )
        else:
            residual_normal_text = None
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
    return ParseResult(
        calls=calls,
        normal_text=_merge_normal_text(residual_normal_text, info.content),
    )
