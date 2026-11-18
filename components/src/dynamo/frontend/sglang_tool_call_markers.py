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
import re
from functools import lru_cache
from typing import Any

from sglang.srt.function_call.function_call_parser import FunctionCallParser

from .utils import PreprocessError

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


# A content span is swapped for ``\ue100<index>\ue101`` while the template is
# rendered a second time; the two renderings are then aligned piece by piece.
# Content that itself contains either character is treated as a span, so the
# sentinel cannot be forged from the request.
_CONTENT_SPAN_SENTINEL_RE = re.compile("\ue100(\\d+)\ue101")
_CONTENT_SPAN_SENTINEL_CHARS = ("\ue100", "\ue101")


def _isolate_marker_spans(
    messages: list[dict[str, Any]], markers: tuple[str, ...]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Replace every content string that contains a delimiter with a sentinel.

    Returns the substituted messages and the original spans, indexed by the
    number embedded in each sentinel. Content without a delimiter is left as
    is, so its tokenization is byte-for-byte unchanged.

    Templates that keep list content render adjacent text parts back to back,
    so a delimiter may only exist in their concatenation (``"<tool_"`` +
    ``"call>"``). When the joined text of a message's parts contains one,
    every non-empty text part of that message becomes its own span; spans are
    tokenized one at a time, so no token can straddle a part boundary.
    """
    spans: list[str] = []
    needles = markers + _CONTENT_SPAN_SENTINEL_CHARS

    def has_needle(text: str) -> bool:
        return any(n in text for n in needles)

    def substitute(text: str, force: bool = False) -> str:
        if not text or not (force or has_needle(text)):
            return text
        spans.append(text)
        return f"\ue100{len(spans) - 1}\ue101"

    def is_text_part(item: Any) -> bool:
        return (
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        )

    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            new_content: Any = substitute(content)
        elif isinstance(content, list):
            joined = "".join(item["text"] for item in content if is_text_part(item))
            force = has_needle(joined)
            new_content = []
            for item in content:
                if is_text_part(item):
                    item = {**item, "text": substitute(item["text"], force)}
                new_content.append(item)
        else:
            new_content = content
        out.append(
            {**msg, "content": new_content} if new_content is not content else msg
        )
    return out, spans


def _align_spans(
    rendered: str, skeleton: str, spans: list[str]
) -> list[tuple[str, bool]] | None:
    """Split *rendered* into ``(text, is_span)`` pieces using the *skeleton*.

    A span may appear with surrounding whitespace stripped (chat templates
    commonly ``strip()`` assistant content), so the stripped variants are
    accepted when the verbatim span does not match. Returns ``None`` when the
    two renderings cannot be reconciled, including a sentinel whose index is
    out of range, so the caller can fail closed.
    """
    pieces: list[tuple[str, bool]] = []
    pos = 0
    for index, part in enumerate(_CONTENT_SPAN_SENTINEL_RE.split(skeleton)):
        if index % 2 == 0:
            if part:
                if not rendered.startswith(part, pos):
                    return None
                pieces.append((part, False))
                pos += len(part)
            continue
        span_index = int(part)
        if span_index >= len(spans):
            return None
        span = spans[span_index]
        for candidate in (span, span.strip(), span.rstrip(), span.lstrip()):
            if candidate and rendered.startswith(candidate, pos):
                pieces.append((candidate, True))
                pos += len(candidate)
                break
        else:
            return None
    return pieces if pos == len(rendered) else None


def encode_chat_prompt_isolating_markers(
    tokenizer: Any,
    template_messages: list[dict[str, Any]],
    template_kwargs: dict[str, Any],
    tool_call_parser_name: str | None,
) -> list[int]:
    """Render and tokenize the chat prompt, keeping delimiters in content as text.

    Content spans that contain a tool-call delimiter are tokenized with the
    delimiter-free encoder; everything the template itself emits (the tool
    instructions, prior assistant ``tool_calls``) keeps the delimiter token
    ids. Prompts whose content has no delimiter tokenize exactly as
    ``apply_chat_template(tokenize=True)`` does.

    Fails closed: when the parser's delimiters are not single tokens for this
    tokenizer, when the spans cannot be aligned with the rendered prompt, or
    when the tokenizer cannot be copied, the request is rejected rather than
    encoded with the delimiter ids.
    """
    strings = tool_call_marker_strings(tool_call_parser_name, tokenizer)
    marker_ids = tool_call_marker_token_ids(tokenizer, tool_call_parser_name)
    if strings and not marker_ids:
        # The parser has delimiters but this tokenizer does not give them
        # single ids, so neither the output gate nor isolation can tell a
        # genuine delimiter from a spelled one. Reject rather than pretend.
        raise PreprocessError(
            "--isolate-tool-call-markers is enabled but the tool-call delimiters "
            f"{strings!r} of parser {tool_call_parser_name!r} are not single "
            "tokens for this tokenizer"
        )
    markers: tuple[str, ...] = strings if strings and marker_ids else ()
    substituted, spans = (
        _isolate_marker_spans(template_messages, markers) if markers else ([], [])
    )
    if not spans:
        rendered = tokenizer.apply_chat_template(template_messages, **template_kwargs)
        ids = getattr(rendered, "input_ids", rendered)
        if isinstance(ids, dict):
            ids = ids.get("input_ids", rendered)
        return list(ids)

    _reject_unsupported_tokenizer_controls(template_kwargs)
    text_kwargs = {**template_kwargs, "tokenize": False}
    rendered = tokenizer.apply_chat_template(template_messages, **text_kwargs)
    skeleton = tokenizer.apply_chat_template(substituted, **text_kwargs)
    pieces = _align_spans(rendered, skeleton, spans)
    if pieces is None:
        logger.warning(
            "Rejecting request: the chat template rewrote a content span that "
            "contains a tool-call delimiter, so it cannot be isolated"
        )
        raise PreprocessError(
            "Message content containing tool-call delimiters could not be encoded "
            "safely for this chat template."
        )
    encoder = _marker_free_encoder(tokenizer, markers, set(marker_ids))
    if encoder is None:
        logger.error(
            "Rejecting request: no delimiter-free tokenizer copy is available, so "
            "tool-call delimiters in message content cannot be isolated"
        )
        raise PreprocessError(
            "Message content containing tool-call delimiters is not supported by "
            "this tokenizer."
        )

    prompt_token_ids: list[int] = []
    for text, is_span in pieces:
        if is_span:
            prompt_token_ids.extend(encoder.encode(text, add_special_tokens=False).ids)
        else:
            prompt_token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    return _truncate_prompt_token_ids(tokenizer, prompt_token_ids, template_kwargs)


# apply_chat_template(tokenize=True) forwards these to tokenizer.__call__; the
# isolated path renders with tokenize=False, so it must apply them itself.
_UNSUPPORTED_TOKENIZER_CONTROLS = ("padding", "tokenizer_kwargs", "return_tensors")


def _reject_unsupported_tokenizer_controls(template_kwargs: dict[str, Any]) -> None:
    """Fail closed on tokenizer controls that would change the assembled ids."""
    active = [
        key
        for key in _UNSUPPORTED_TOKENIZER_CONTROLS
        if template_kwargs.get(key) not in (None, False, "do_not_pad", {})
    ]
    if active:
        raise PreprocessError(
            f"chat_template_kwargs {active!r} are not supported for message content "
            "that contains tool-call delimiters while --isolate-tool-call-markers "
            "is enabled"
        )


def _truncate_prompt_token_ids(
    tokenizer: Any, prompt_token_ids: list[int], template_kwargs: dict[str, Any]
) -> list[int]:
    """Apply ``truncation``/``max_length`` as ``tokenizer.__call__`` would."""
    truncation = template_kwargs.get("truncation")
    if truncation not in (True, "longest_first", "only_first"):
        return prompt_token_ids
    limit = template_kwargs.get("max_length") or getattr(
        tokenizer, "model_max_length", None
    )
    if not limit or len(prompt_token_ids) <= limit:
        return prompt_token_ids
    if getattr(tokenizer, "truncation_side", "right") == "left":
        return prompt_token_ids[-limit:]
    return prompt_token_ids[:limit]
