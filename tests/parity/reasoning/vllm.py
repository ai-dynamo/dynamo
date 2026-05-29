# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning parity wrapper for vLLM's reasoning parser classes."""

from __future__ import annotations

import re
from typing import Any

from vllm.entrypoints.openai.parser.harmony_utils import get_encoding, parse_chat_output
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers.mistral import MistralTokenizer

from tests.parity.common import _FAMILY_TO_VLLM_REASONING, ReasoningResult

# Harmony text-level fallback regexes. Mirror tests/parity/toolcalling/vllm.py so the
# behavior matches when get_encoding() can't load the tiktoken vocab (e.g. CI
# runners with no reach to openaipublic.blob.core.windows.net).
_HARMONY_MESSAGE_RE = re.compile(
    r"(?:<\|start\|>assistant)?"
    r"<\|channel\|>(?P<channel>\w+)"
    r"(?P<header>.*?)"
    r"<\|message\|>(?P<body>.*?)"
    r"(?P<stop><\|call\|>|<\|end\|>|<\|return\|>|$)",
    re.DOTALL,
)
_HARMONY_RECIPIENT_RE = re.compile(r"\bto=(?P<recipient>functions\.[\w.\-]+)")
_HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"


class _HarmonyEncodingUnavailable(RuntimeError):
    """Raised when openai-harmony's tiktoken vocab can't be loaded.

    The Rust extension behind openai-harmony downloads `o200k_base.tiktoken`
    from openaipublic.blob.core.windows.net on first use. CI runners without
    reach to that CDN fail with HarmonyError. The reasoning fixtures fall back
    to a text-level harmony state machine when this happens — mirrors what
    tests/parity/toolcalling/vllm.py already does for the tool-calling parity path.
    """


_SPECIAL_TOKEN_IDS = {
    "<think>": 1_000_001,
    "</think>": 1_000_002,
    "<tool_call>": 1_000_003,
    "</tool_call>": 1_000_004,
    "<|channel>": 1_000_005,
    "<channel|>": 1_000_006,
    "<|turn>": 1_000_007,
    "<|tool_call>": 1_000_008,
    "<|tool_response>": 1_000_009,
    "[THINK]": 1_000_010,
    "[/THINK]": 1_000_011,
    "<|start|>": 1_000_012,
    "<|channel|>": 1_000_013,
    "<|constrain|>": 1_000_014,
    "<|message|>": 1_000_015,
    "<|end|>": 1_000_016,
    "<|call|>": 1_000_017,
    "<|return|>": 1_000_018,
    "<|tool_calls_section_begin|>": 1_000_019,
}
_ID_TO_SPECIAL_TOKEN = {v: k for k, v in _SPECIAL_TOKEN_IDS.items()}
_SPECIAL_TOKENS_BY_LENGTH = sorted(
    _SPECIAL_TOKEN_IDS,
    key=len,
    reverse=True,
)

_HARMONY_SPECIAL_TOKENS = (
    "<|channel|>",
    "<|constrain|>",
    "<|end|>",
    "<|message|>",
    "<|start|>",
    "<|call|>",
    "<|return|>",
)


class _StubTokenizer:
    all_special_tokens = tuple(_SPECIAL_TOKEN_IDS)

    def get_vocab(self) -> dict[str, int]:
        return dict(_SPECIAL_TOKEN_IDS)

    def encode(self, text: str, *args: Any, **kwargs: Any) -> list[int]:
        token_ids: list[int] = []
        i = 0
        while i < len(text):
            for token in _SPECIAL_TOKENS_BY_LENGTH:
                if text.startswith(token, i):
                    token_ids.append(_SPECIAL_TOKEN_IDS[token])
                    i += len(token)
                    break
            else:
                token_ids.append(ord(text[i]))
                i += 1
        return token_ids

    def decode(self, token_ids: list[int], *args: Any, **kwargs: Any) -> str:
        return "".join(_ID_TO_SPECIAL_TOKEN.get(t, chr(t)) for t in token_ids)


class _HarmonyTokenizer:
    def __init__(self) -> None:
        try:
            self.encoding = get_encoding()
        except Exception as e:
            raise _HarmonyEncodingUnavailable(f"{type(e).__name__}: {e}") from e

    def encode(self, text: str, *args: Any, **kwargs: Any) -> list[int]:
        return list(self.encoding.encode(text, allowed_special="all"))

    def decode(self, token_ids: list[int], *args: Any, **kwargs: Any) -> str:
        return self.encoding.decode(token_ids)

    def get_vocab(self) -> dict[str, int]:
        return {token: self.encode(token)[0] for token in _HARMONY_SPECIAL_TOKENS}


class _StubMistralInnerTokenizer:
    def get_special_token(self, token: Any) -> int | None:
        key = getattr(token, "value", token)
        return _SPECIAL_TOKEN_IDS.get(str(key))


class _StubMistralTokenizer(MistralTokenizer):
    all_special_tokens = tuple(_SPECIAL_TOKEN_IDS)

    def __init__(self) -> None:
        self.tokenizer = _StubMistralInnerTokenizer()

    def get_vocab(self) -> dict[str, int]:
        return dict(_SPECIAL_TOKEN_IDS)

    def encode(self, text: str, *args: Any, **kwargs: Any) -> list[int]:
        return _StubTokenizer().encode(text, *args, **kwargs)

    def decode(self, token_ids: list[int], *args: Any, **kwargs: Any) -> str:
        return _StubTokenizer().decode(token_ids, *args, **kwargs)

    def __len__(self) -> int:
        return len(_SPECIAL_TOKEN_IDS)


class _Request:
    skip_special_tokens = False


def _make_parser(parser_name: str, fixture: dict[str, Any]) -> Any:
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    if parser_name == "openai_gptoss":
        tokenizer = _HarmonyTokenizer()
    elif parser_name == "mistral":
        tokenizer = _StubMistralTokenizer()
    else:
        tokenizer = _StubTokenizer()
    chat_template_kwargs = {}
    if parser_name == "deepseek_v4":
        chat_template_kwargs["enable_thinking"] = True
    if parser_name == "deepseek_v3":
        chat_template_kwargs["thinking"] = True
    chat_template_kwargs.update(fixture.get("chat_template_kwargs", {}))

    attempts = (
        lambda: parser_cls(tokenizer, chat_template_kwargs=chat_template_kwargs),
        lambda: parser_cls(tokenizer),
        lambda: parser_cls(),
    )
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as e:
            last_error = e
    raise TypeError(f"could not initialize vLLM reasoning parser: {last_error}")


def _message_field(message: Any, *names: str) -> str:
    if message is None:
        return ""
    for name in names:
        if isinstance(message, dict):
            value = message.get(name)
        else:
            value = getattr(message, name, None)
        if value:
            return value
    return ""


def _run_stream(
    parser: Any, fixture: dict[str, Any], chunks: list[str]
) -> ReasoningResult:
    token_chunks = fixture.get("token_chunks")
    previous_text = ""
    previous_token_ids: list[int] = []
    reasoning_text = ""
    normal_text = ""
    reasoning_done = False
    tokenizer = parser.model_tokenizer

    for i, chunk in enumerate(chunks):
        current_text = previous_text + chunk
        current_token_ids = tokenizer.encode(current_text)
        if token_chunks:
            delta_token_ids = token_chunks[i]
        elif current_token_ids[: len(previous_token_ids)] == previous_token_ids:
            delta_token_ids = current_token_ids[len(previous_token_ids) :]
        else:
            delta_token_ids = tokenizer.encode(chunk)

        if reasoning_done:
            normal_text += chunk
        else:
            message = parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                chunk,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )
            reasoning_text += _message_field(message, "reasoning", "reasoning_content")
            normal_text += _message_field(message, "content")
            if parser.is_reasoning_end_streaming(current_token_ids, delta_token_ids):
                reasoning_done = True

        previous_text = current_text
        previous_token_ids = current_token_ids

    return ReasoningResult(reasoning_text=reasoning_text, normal_text=normal_text)


def _run_batch(parser: Any, fixture: dict[str, Any]) -> ReasoningResult:
    reasoning_text, normal_text = parser.extract_reasoning(
        fixture["model_text"],
        _Request(),
    )
    return ReasoningResult(reasoning_text=reasoning_text, normal_text=normal_text)


def _run_gptoss_batch(fixture: dict[str, Any]) -> ReasoningResult:
    try:
        token_ids = _HarmonyTokenizer().encode(fixture["model_text"])
        reasoning_text, normal_text, _ = parse_chat_output(token_ids)
    except _HarmonyEncodingUnavailable:
        return _harmony_reasoning_parse_batch(fixture["model_text"])
    except Exception as e:
        # parse_chat_output() reaches into get_encoding() too — raw HarmonyError
        # surfaces here, not _HarmonyEncodingUnavailable. Catch and route the
        # same way so the fallback runs.
        if _looks_like_harmony_vocab_error(e):
            return _harmony_reasoning_parse_batch(fixture["model_text"])
        raise
    return ReasoningResult(reasoning_text=reasoning_text, normal_text=normal_text)


def _looks_like_harmony_vocab_error(exc: Exception) -> bool:
    msg = str(exc)
    return "vocab file" in msg or "tiktoken" in msg


def _harmony_parse_messages(text: str) -> tuple[list[str], list[str]]:
    """Return (analysis_bodies, normal_bodies) extracted from harmony markup.

    Mirrors openai-harmony's parse_chat_output() at the text level:
    - `analysis` channel bodies → reasoning_text contributors
    - `final` channel bodies → normal_text contributors
    - `commentary` channel bodies WITHOUT `to=functions.*` → normal_text contributors
    - `commentary` channel bodies WITH a `to=functions.*` recipient → dropped
      (tool-call payload; the tool-call parser handles it separately)
    """
    analysis_bodies: list[str] = []
    normal_bodies: list[str] = []
    for match in _HARMONY_MESSAGE_RE.finditer(text):
        channel = match.group("channel")
        header = match.group("header")
        body = match.group("body")
        recipient_match = _HARMONY_RECIPIENT_RE.search(header)
        recipient = recipient_match.group("recipient") if recipient_match else None
        if channel == "analysis":
            analysis_bodies.append(body)
        elif channel == "final":
            normal_bodies.append(body)
        elif channel == "commentary" and recipient is None:
            normal_bodies.append(body)
    return analysis_bodies, normal_bodies


def _harmony_reasoning_parse_batch(model_text: str) -> ReasoningResult:
    """Text-level mirror of parse_chat_output() for batch mode."""
    analysis_bodies, normal_bodies = _harmony_parse_messages(model_text)
    return ReasoningResult(
        reasoning_text="\n".join(analysis_bodies),
        normal_text="\n".join(normal_bodies),
    )


def _harmony_reasoning_parse_stream(chunks: list[str]) -> ReasoningResult:
    """Text-level mirror of extract_reasoning_streaming() across all chunks.

    Re-parses the accumulated text on every chunk and emits the delta vs the
    previous chunk's parse. Once we've passed `<|channel|>final<|message|>`
    the harness convention is to raw-append subsequent chunks to normal_text
    (matches what _run_stream does when is_reasoning_end_streaming flips).
    """
    reasoning_text = ""
    normal_text = ""
    accumulated = ""
    prev_reasoning = ""
    prev_normal = ""
    reasoning_done = False
    for chunk in chunks:
        if reasoning_done:
            normal_text += chunk
            continue
        accumulated += chunk
        analysis_bodies, normal_bodies = _harmony_parse_messages(accumulated)
        cur_reasoning = "\n".join(analysis_bodies)
        cur_normal = "\n".join(normal_bodies)
        if cur_reasoning.startswith(prev_reasoning):
            reasoning_text += cur_reasoning[len(prev_reasoning) :]
        else:
            reasoning_text = cur_reasoning
        if cur_normal.startswith(prev_normal):
            normal_text += cur_normal[len(prev_normal) :]
        else:
            normal_text = cur_normal
        prev_reasoning = cur_reasoning
        prev_normal = cur_normal
        if _HARMONY_FINAL_MARKER in accumulated:
            reasoning_done = True
    return ReasoningResult(reasoning_text=reasoning_text, normal_text=normal_text)


def parse(
    parser_family: str,
    fixture: dict[str, Any],
    mode: str,
) -> ReasoningResult:
    parser_name = fixture.get("vllm_parser") or _FAMILY_TO_VLLM_REASONING.get(
        parser_family
    )
    if not parser_name:
        return ReasoningResult(
            error=(
                "UNAVAILABLE: vLLM has no reasoning parser for "
                f"family={parser_family!r}"
            )
        )

    try:
        if parser_name == "openai_gptoss" and mode == "batch":
            return _run_gptoss_batch(fixture)
        try:
            parser = _make_parser(parser_name, fixture)
        except _HarmonyEncodingUnavailable:
            if parser_name == "openai_gptoss" and mode == "stream":
                return _harmony_reasoning_parse_stream(fixture["chunks"])
            raise
        if mode == "stream":
            chunks = fixture["chunks"]
            return _run_stream(parser, fixture, chunks)
        elif mode == "batch":
            return _run_batch(parser, fixture)
        else:
            raise ValueError(f"unsupported reasoning mode: {mode!r}")
    except _HarmonyEncodingUnavailable:
        if parser_name == "openai_gptoss" and mode == "batch":
            return _harmony_reasoning_parse_batch(fixture["model_text"])
        if parser_name == "openai_gptoss" and mode == "stream":
            return _harmony_reasoning_parse_stream(fixture["chunks"])
        raise
    except Exception as e:
        if parser_name == "openai_gptoss" and _looks_like_harmony_vocab_error(e):
            if mode == "batch":
                return _harmony_reasoning_parse_batch(fixture["model_text"])
            if mode == "stream":
                return _harmony_reasoning_parse_stream(fixture["chunks"])
        return ReasoningResult(error=f"{type(e).__name__}: {e}")
