# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared markup colorization helpers for parity table tooltips."""

from __future__ import annotations

import html as html_lib
import re
from typing import Any

# Matches both `<...>` and the Mistral-style `[NAME]`/`[/NAME]` form.
# Brackets only match ALL-CAPS-underscore names so JSON arrays
# (e.g. `[{...}]`, `[1, 2]`) don't get false-matched as tags.
_TAG_RE = re.compile(r"<[^<>]+>|\[/?[A-Z][A-Z0-9_]*\]")

# Two coloring schemes share one cycling palette:
#   * Paired open/close: every matched pair gets a *fresh* palette index, so
#     two `<tool_call>...</tool_call>` instances in the same input render as
#     two different colors.
#   * Singletons (e.g. harmony's `<|channel|>` section markers): stable
#     by-role color so all `<|channel|>` tokens share one class everywhere.
_PAIRED_PALETTE_SIZE = 8
_color_seq = 0
_singleton_classes: dict[str, str] = {}


def _next_color_class() -> str:
    global _color_seq
    cls = f"tt-c{_color_seq % _PAIRED_PALETTE_SIZE}"
    _color_seq += 1
    return cls


def _singleton_class_for(name: str) -> str:
    cls = _singleton_classes.get(name)
    if cls is None:
        cls = _next_color_class()
        _singleton_classes[name] = cls
    return cls


_PIPES = ("|", "｜")  # ASCII and FULLWIDTH VERTICAL LINE (U+FF5C)
_BEGIN_SUFFIXES = (
    "_begin",
    "▁begin",
)  # ASCII underscore and LOWER ONE EIGHTH BLOCK (U+2581)
_END_SUFFIXES = ("_end", "▁end")

# Harmony (gpt-oss) tokens are linear state-machine delimiters. Turn-boundary
# tokens form pseudo-pairs: `<|start|>` opens a turn; `<|end|>`, `<|return|>`,
# or `<|call|>` closes it. Each closer flavor gets its own paired color so a
# tool call turn (start -> call) is visually distinct from a normal turn
# (start -> end) or a final turn (start -> return). Section markers stay as
# same-colored singletons.
_HARMONY_TURN_OPEN = "start"
_HARMONY_TURN_CLOSE = frozenset({"end", "return", "call"})
_HARMONY_SECTION_MARKERS = frozenset({"channel", "constrain", "message"})
_HARMONY_TOKEN_RE = re.compile(r"<\|([A-Za-z_]+)\|>")
_HARMONY_SEGMENT_CLASS = {
    "start": "tt-h-start",
    "channel": "tt-h-channel",
    "constrain": "tt-h-constrain",
    "message": "tt-h-message",
    "end": "tt-h-stop",
    "return": "tt-h-stop",
    "call": "tt-h-call",
}


def _colorize_harmony(text: str) -> str:
    """Color Harmony's linear token segments as `<|token|>related-text`.

    Harmony is not paired XML. Tokens such as `<|channel|>` and `<|message|>`
    introduce the header/content span that follows, so each special token is
    grouped with text up to the next Harmony token.
    """
    pieces: list[str] = []
    last = 0
    for m in _HARMONY_TOKEN_RE.finditer(text):
        if m.start() > last:
            pieces.append(html_lib.escape(text[last : m.start()]))
        token_name = m.group(1)
        if token_name in _HARMONY_TURN_CLOSE:
            end = m.end()
        else:
            next_m = _HARMONY_TOKEN_RE.search(text, m.end())
            end = next_m.start() if next_m else len(text)
        cls = _HARMONY_SEGMENT_CLASS.get(token_name, "tt-h-other")
        token = html_lib.escape(m.group(0))
        related = html_lib.escape(text[m.end() : end])
        pieces.append(
            f'<span class="tt-h {cls}"><span class="tt-h-token">{token}</span>{related}</span>'
        )
        last = end
    if last < len(text):
        pieces.append(html_lib.escape(text[last:]))
    return "".join(pieces)


def colorize_markup(text: str, family: str | None = None) -> str:
    if family == "harmony" and _HARMONY_TOKEN_RE.search(text):
        return _colorize_harmony(text)
    return _colorize_xml(text)


def _strip_suffix(s: str, suffixes: tuple[str, ...]) -> str | None:
    for suf in suffixes:
        if s.endswith(suf) and len(s) > len(suf):
            return s[: -len(suf)]
    return None


def _tag_kind_and_name(inner: str) -> tuple[str | None, str, str | None]:
    """Classify `<...>` tag inner text into an open/close/singleton kind.

    Returns (kind, pair_id, color_override):
      * kind: 'open' | 'close' | 'singleton' | 'toggle' | None
      * pair_id: name used to match open against close on the stack
      * color_override: when set, the paired span uses this name for the
        color class instead of pair_id (lets `<|start|>...<|call|>` color
        differently from `<|start|>...<|end|>` despite sharing pair_id).
    """

    def _name_of(s: str) -> str:
        # Split on whitespace, slash, `>`, OR `=` so that
        # `<function=book_flight>` pairs with `</function>` and
        # `<parameter=destination>` pairs with `</parameter>`.
        if not s:
            return ""
        return re.split(r"[\s/>=]", s, 1)[0].rstrip("|")

    if inner.startswith("/"):
        return ("close", _name_of(inner[1:]), None)
    starts_pipe = inner[:1] in _PIPES
    ends_pipe = inner[-1:] in _PIPES
    if starts_pipe and ends_pipe and len(inner) >= 2:
        middle = inner[1:-1]
        stripped = _strip_suffix(middle, _BEGIN_SUFFIXES)
        if stripped is not None:
            return ("open", stripped, None)
        stripped = _strip_suffix(middle, _END_SUFFIXES)
        if stripped is not None:
            return ("close", stripped, None)
        if middle == _HARMONY_TURN_OPEN:
            return ("open", "__harmony_turn", None)
        if middle in _HARMONY_TURN_CLOSE:
            # Color the pair by which closer flavor was used.
            return ("close", "__harmony_turn", f"__harmony_pair_{middle}")
        if middle in _HARMONY_SECTION_MARKERS:
            return ("singleton", "__harmony_section", None)
        # gemma4 string delimiter `<|"|>` is a self-paired quote token: the
        # same literal both opens and closes the string. Classify it as
        # `toggle` so a matched `<|"|>value<|"|>` pair colors like any other
        # pair, while a dangling (truncated) delimiter still falls through to
        # orphan/red. The open-vs-close decision is made by the stack at use
        # site, since the literal alone cannot say which it is.
        if middle == '"':
            return ("toggle", "__gemma_quote", None)
        return (None, "", None)
    if starts_pipe and inner[:1] == "|":
        return ("open", _name_of(inner[1:]), None)
    if ends_pipe and inner[-1:] == "|":
        return ("close", _name_of(inner[:-1]), None)
    return ("open", _name_of(inner), None)


def _colorize_xml(text: str) -> str:
    """HTML-escape `text` and wrap each `<...>` token in a span.
    Paired open/close (stack match by tag name) -> class 'tt-paired'.
    Unmatched close, or open that never closes -> class 'tt-orphan'.

    Pairs standard XML (`<X>...</X>`) AND alt pipe-marker conventions:
      `<|X>...<X|>`          (boundary ASCII pipes)
      `<|X_begin|>...<|X_end|>`  (both-side pipes with _begin/_end suffix)
    Lenient pop-through: a close looks down the stack for the nearest
    name-match; anything un-closed above it is marked orphan. Lets
    no-close singletons (e.g. `<|tool_call_argument_begin|>`) localize
    their orphan-ness without poisoning the surrounding pairs.
    """
    pieces: list[str] = []
    stack: list[tuple[str, int]] = []
    last = 0
    for m in _TAG_RE.finditer(text):
        if m.start() > last:
            pieces.append(html_lib.escape(text[last : m.start()]))
        tok = m.group(0)
        kind, pair_id, color_override = _tag_kind_and_name(tok[1:-1])
        esc = html_lib.escape(tok)
        if kind is None:
            pieces.append(f'<span class="tt-orphan">{esc}</span>')
        elif kind == "singleton":
            cls = _singleton_class_for(pair_id)
            pieces.append(f'<span class="{cls}">{esc}</span>')
        elif kind == "toggle":
            # Self-paired delimiter: close the nearest matching open if one is
            # on the stack, otherwise treat this as the open. A leftover open
            # is orphaned by the end-of-loop cleanup.
            match_at = -1
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == pair_id:
                    match_at = i
                    break
            if match_at >= 0:
                for _, unmatched_idx in stack[match_at + 1 :]:
                    pieces[
                        unmatched_idx
                    ] = f'<span class="tt-orphan">{pieces[unmatched_idx]}</span>'
                open_idx = stack[match_at][1]
                cls = _next_color_class()
                pieces[open_idx] = f'<span class="{cls}">{pieces[open_idx]}</span>'
                pieces.append(f'<span class="{cls}">{esc}</span>')
                del stack[match_at:]
            else:
                pieces.append(esc)
                stack.append((pair_id, len(pieces) - 1))
        elif kind == "close":
            match_at = -1
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == pair_id:
                    match_at = i
                    break
            if match_at >= 0:
                for _, unmatched_idx in stack[match_at + 1 :]:
                    pieces[
                        unmatched_idx
                    ] = f'<span class="tt-orphan">{pieces[unmatched_idx]}</span>'
                open_idx = stack[match_at][1]
                # Per-instance color: every matched pair gets a fresh palette
                # index, so two `<tool_call>...</tool_call>` (or two
                # `<|start|>...<|call|>`) blocks in the same input render as
                # different colors. (color_override unused on the paired
                # path — kept on the singleton-flavor branch only.)
                _ = color_override
                cls = _next_color_class()
                pieces[open_idx] = f'<span class="{cls}">{pieces[open_idx]}</span>'
                pieces.append(f'<span class="{cls}">{esc}</span>')
                del stack[match_at:]
            else:
                pieces.append(f'<span class="tt-orphan">{esc}</span>')
        else:
            pieces.append(esc)
            stack.append((pair_id, len(pieces) - 1))
        last = m.end()
    for _, idx in stack:
        pieces[idx] = f'<span class="tt-orphan">{pieces[idx]}</span>'
    if last < len(text):
        pieces.append(html_lib.escape(text[last:]))
    return "".join(pieces)


def _xml_token_intervals(text: str) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    stack: list[tuple[str, int]] = []
    for match in _TAG_RE.finditer(text):
        idx = len(intervals)
        tok = match.group(0)
        kind, pair_id, _color_override = _tag_kind_and_name(tok[1:-1])
        intervals.append({"start": match.start(), "end": match.end(), "class": None})
        if kind is None:
            intervals[idx]["class"] = "tt-orphan"
        elif kind == "singleton":
            intervals[idx]["class"] = _singleton_class_for(pair_id)
        elif kind == "toggle":
            match_at = -1
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == pair_id:
                    match_at = i
                    break
            if match_at >= 0:
                for _, unmatched_idx in stack[match_at + 1 :]:
                    intervals[unmatched_idx]["class"] = "tt-orphan"
                cls = _next_color_class()
                intervals[stack[match_at][1]]["class"] = cls
                intervals[idx]["class"] = cls
                del stack[match_at:]
            else:
                stack.append((pair_id, idx))
        elif kind == "close":
            match_at = -1
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == pair_id:
                    match_at = i
                    break
            if match_at >= 0:
                for _, unmatched_idx in stack[match_at + 1 :]:
                    intervals[unmatched_idx]["class"] = "tt-orphan"
                cls = _next_color_class()
                intervals[stack[match_at][1]]["class"] = cls
                intervals[idx]["class"] = cls
                del stack[match_at:]
            else:
                intervals[idx]["class"] = "tt-orphan"
        else:
            stack.append((pair_id, idx))
    for _, idx in stack:
        intervals[idx]["class"] = "tt-orphan"
    return intervals


def _harmony_intervals(text: str) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    for match in _HARMONY_TOKEN_RE.finditer(text):
        token_name = match.group(1)
        if token_name in _HARMONY_TURN_CLOSE:
            end = match.end()
        else:
            next_match = _HARMONY_TOKEN_RE.search(text, match.end())
            end = next_match.start() if next_match else len(text)
        intervals.append(
            {
                "start": match.start(),
                "end": end,
                "class": _HARMONY_SEGMENT_CLASS.get(token_name, "tt-h-other"),
                "token_start": match.start(),
                "token_end": match.end(),
                "harmony": True,
            }
        )
    return intervals


def _markup_intervals(text: str, family: str | None) -> list[dict[str, Any]]:
    if family == "harmony" and _HARMONY_TOKEN_RE.search(text):
        return _harmony_intervals(text)
    return _xml_token_intervals(text)


def _render_harmony_interval_slice(
    text: str, interval: dict[str, Any], start: int, end: int
) -> str:
    token_start = interval["token_start"]
    token_end = interval["token_end"]
    parts: list[str] = []
    cursor = start
    token_slice_start = max(start, token_start)
    token_slice_end = min(end, token_end)
    if cursor < token_slice_start:
        parts.append(html_lib.escape(text[cursor:token_slice_start]))
        cursor = token_slice_start
    if token_slice_start < token_slice_end:
        parts.append(
            '<span class="tt-h-token">'
            f"{html_lib.escape(text[token_slice_start:token_slice_end])}</span>"
        )
        cursor = token_slice_end
    if cursor < end:
        parts.append(html_lib.escape(text[cursor:end]))
    return f'<span class="tt-h {interval["class"]}">{"".join(parts)}</span>'


def _render_interval_slice(
    text: str, interval: dict[str, Any], start: int, end: int
) -> str:
    if interval.get("harmony"):
        return _render_harmony_interval_slice(text, interval, start, end)
    return (
        f'<span class="{interval["class"] or "tt-orphan"}">'
        f"{html_lib.escape(text[start:end])}</span>"
    )


def _colorize_markup_slice(
    text: str, intervals: list[dict[str, Any]], start: int, end: int
) -> str:
    pieces: list[str] = []
    cursor = start
    for interval in intervals:
        istart = interval["start"]
        iend = interval["end"]
        if iend <= start:
            continue
        if istart >= end:
            break
        overlap_start = max(start, istart)
        overlap_end = min(end, iend)
        if cursor < overlap_start:
            pieces.append(html_lib.escape(text[cursor:overlap_start]))
        pieces.append(
            _render_interval_slice(text, interval, overlap_start, overlap_end)
        )
        cursor = overlap_end
    if cursor < end:
        pieces.append(html_lib.escape(text[cursor:end]))
    return "".join(pieces)


def colorize_stream_deltas(chunks: list[Any], family: str | None) -> list[str]:
    deltas = [
        str(chunk.get("delta_text") or "") if isinstance(chunk, dict) else ""
        for chunk in chunks
    ]
    text = "".join(deltas)
    intervals = _markup_intervals(text, family)
    rendered: list[str] = []
    cursor = 0
    for delta in deltas:
        end = cursor + len(delta)
        rendered.append(_colorize_markup_slice(text, intervals, cursor, end))
        cursor = end
    return rendered
