# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-call delimiter tokens of the configured SGLang parser.

Most tool-call formats open and close a call with a delimiter that the
tokenizer registers as a single added token (``<tool_call>`` is 154843 for
GLM-5 and 151657 for Qwen3). Those tokens are not reserved: a literal
``<tool_call>`` inside a tool result or a pasted file encodes to the very
same id the model emits to open a call, and decodes back to identical text.
SGLang's detectors work on text, so on their own they cannot tell a call
the model issued from a delimiter it merely reproduced from its input.

The frontend is the one place that still has the token ids. It uses them in
two ways:

* ``SglangStreamingPostProcessor`` opens the tool-call parser only on the
  delimiter *token id* and closes it on the closing id. Delimiter text the
  model spells out of ordinary sub-words never reaches the parser and is
  returned to the client as content.
* With ``isolate_tool_call_markers`` the prompt encoder tokenizes a message
  content span that contains a delimiter without added-token matching, so
  the model sees ``<``, ``tool``, ``_call``, ``>`` and has no single token
  to copy back. Delimiters emitted by the chat template itself keep their
  ids.
"""

from __future__ import annotations

import inspect
import json
import logging
from functools import lru_cache
from typing import Any

from sglang.srt.function_call.function_call_parser import FunctionCallParser

logger = logging.getLogger(__name__)

_SGLANG_PARSER_NAME_ALIASES = {
    # Dynamo's Rust parser registry accepts these MiniMax-M3 aliases. SGLang
    # builds that include MiniMax-M3 expose the parser as "minimax-m3".
    "minimax_m3": "minimax-m3",
    "minimax_m3_nom": "minimax-m3",
    "minimax-m3-nom": "minimax-m3",
    "kimi-k3": "kimi_k3",
    "gemma-4": "gemma4",
}


def _normalize_sglang_parser_name(parser_name: str | None) -> str | None:
    if not parser_name:
        return parser_name
    return _SGLANG_PARSER_NAME_ALIASES.get(parser_name, parser_name)


@lru_cache(maxsize=64)
def _callable_accepts_kwarg(func: Any, kwarg: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if name == kwarg and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _parser_tokenizer_kwargs(parser: Any, tokenizer: Any) -> dict[str, Any]:
    if tokenizer is not None and _callable_accepts_kwarg(parser, "tokenizer"):
        return {"tokenizer": tokenizer}
    return {}


_MARKER_CACHE_ATTR = "_dynamo_tool_call_marker_cache"


def tool_call_marker_strings(
    tool_call_parser_name: str | None, tokenizer: Any = None
) -> tuple[str, str] | None:
    """``(opening, closing)`` delimiter of the named SGLang detector.

    The detector is built the way ``FunctionCallParser`` builds the active one,
    with *tokenizer* when its constructor accepts it: Hunyuan resolves suffixed
    delimiters (``<tool_calls:opensource>``) from the vocabulary, and a bare
    construction would name delimiters the model never emits. Surrounding
    whitespace is dropped (``qwen25`` declares ``<tool_call>\\n``); detectors
    with an empty delimiter on either side return ``None``.
    """
    name = _normalize_sglang_parser_name(tool_call_parser_name)
    if not name:
        return None
    detector_cls = FunctionCallParser.ToolCallParserEnum.get(name)
    if detector_cls is None:
        return None
    cache = _tokenizer_marker_cache(tokenizer) if tokenizer is not None else None
    key = f"strings:{name}"
    if cache is not None and key in cache:
        return cache[key]
    try:
        detector = detector_cls(**_parser_tokenizer_kwargs(detector_cls, tokenizer))
    except Exception:  # pragma: no cover - detector-specific constructor
        # Falling back to text-only parsing here would silently drop the
        # delimiter-id gate, so surface the failure instead.
        logger.exception("Could not construct tool-call detector for parser %s", name)
        raise
    opening = getattr(detector, "bot_token", "")
    closing = getattr(detector, "eot_token", "")
    strings: tuple[str, str] | None = None
    if isinstance(opening, str) and isinstance(closing, str):
        opening, closing = opening.strip(), closing.strip()
        if opening and closing and opening != closing:
            strings = (opening, closing)
    if cache is not None:
        cache[key] = strings
    return strings


def _tokenizer_marker_cache(tokenizer: Any) -> dict[str, Any] | None:
    cache = getattr(tokenizer, _MARKER_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        try:
            setattr(tokenizer, _MARKER_CACHE_ATTR, cache)
        except (AttributeError, TypeError):
            return None
    return cache


def tool_call_marker_token_ids(
    tokenizer: Any, tool_call_parser_name: str | None
) -> dict[int, str]:
    """Map delimiter token id -> delimiter string for the active parser.

    Both delimiters must encode to exactly one token; otherwise the result is
    ``{}``, the frontend keeps the text-only parser behaviour, and
    ``encode_chat_prompt_isolating_markers`` rejects requests instead of
    encoding them unprotected.
    """
    strings = tool_call_marker_strings(tool_call_parser_name, tokenizer)
    if strings is None:
        return {}
    cache = _tokenizer_marker_cache(tokenizer)
    key = f"ids:{tool_call_parser_name}"
    if cache is not None and key in cache:
        return cache[key]
    ids: dict[int, str] = {}
    for marker in strings:
        try:
            encoded = tokenizer.encode(marker, add_special_tokens=False)
        except Exception:
            logger.exception(
                "Could not encode tool-call delimiter %r for parser %s",
                marker,
                tool_call_parser_name,
            )
            continue
        encoded = list(getattr(encoded, "ids", encoded))
        if len(encoded) == 1 and isinstance(encoded[0], int):
            ids[encoded[0]] = marker
    if len(ids) != 2:
        logger.warning(
            "Tool-call delimiters %r of parser %s are not single tokens for this "
            "tokenizer; the delimiter-id gate is off and "
            "--isolate-tool-call-markers cannot be honoured",
            strings,
            tool_call_parser_name,
        )
        ids = {}
    if cache is not None:
        cache[key] = ids
    return ids


def _marker_free_encoder(
    tokenizer: Any, markers: tuple[str, ...], active_marker_ids: set[int]
) -> Any | None:
    """A copy of the fast tokenizer with the delimiter added-tokens removed.

    Encoding a literal delimiter with it yields ordinary BPE sub-words that
    decode back to the same text, so the model sees the delimiter as content
    rather than as structure. ``None`` when the tokenizer cannot be copied, or
    when the copy still encodes a delimiter to one of *active_marker_ids*
    (the base vocabulary can reach the same id without the added-token entry).
    """
    cache = _tokenizer_marker_cache(tokenizer)
    key = "encoder:" + "|".join(markers)
    if cache is not None and key in cache:
        return cache[key]
    encoder = None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        try:
            from tokenizers import Tokenizer as HfTokenizer

            spec = json.loads(backend.to_str())
            spec["added_tokens"] = [
                tok
                for tok in spec.get("added_tokens", [])
                if tok.get("content") not in markers
            ]
            encoder = HfTokenizer.from_str(json.dumps(spec))
            for marker in markers:
                copied = encoder.encode(marker, add_special_tokens=False).ids
                if any(token_id in active_marker_ids for token_id in copied):
                    raise ValueError(
                        f"delimiter {marker!r} still encodes to an active delimiter id"
                    )
        except Exception:
            logger.exception("Could not build a delimiter-free tokenizer copy")
            encoder = None
    if cache is not None:
        cache[key] = encoder
    return encoder
