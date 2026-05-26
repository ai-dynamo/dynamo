# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contract types + canonical-JSON diff for parity (parser) impls.

Every impl wrapper (parser-mode and e2e-mode) returns ParseResult so the
harness can diff results uniformly.
"""

from __future__ import annotations

import html as html_lib
import json
import re
from dataclasses import dataclass, field
from typing import Any

URL_RE = re.compile(r"https?://[^\s<>'\"]+")
TRAILING_URL_PUNCTUATION = ".,;:)"

# Curated featured-model order shared by the parser and reasoning parity tables.
# The row labels still come from fixture metadata; this list only decides which
# tool calling families lead the table and in what order.
TOP_N_TOOL_CALLING_FAMILIES = [
    "deepseek_v4",
    "gemma4",
    "glm47",
    "harmony",
    "kimi_k2",
    "minimax_m2",
    "qwen3_coder",
]


@dataclass
class ParseResult:
    """Uniform shape returned by every impl wrapper.

    `calls` is a list of {"name": str, "arguments": dict}.  Argument values
    are dicts (parsed from JSON) so canonical comparison ignores whitespace
    differences in the wire encoding.
    """

    calls: list[dict[str, Any]] = field(default_factory=list)
    normal_text: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "normal_text": self.normal_text,
            "error": self.error,
        }


@dataclass
class ReasoningResult:
    """Uniform shape returned by every reasoning impl wrapper."""

    reasoning_text: str | None = None
    normal_text: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_text": self.reasoning_text,
            "normal_text": self.normal_text,
            "error": self.error,
        }


def _normalize_normal_text(v: Any) -> Any:
    """Treat any whitespace-only or None value as equivalent for comparison.

    Engines disagree on the carrier for "no narration":
      * Dynamo emits ``""`` (or ``"\\n"`` carried through between back-to-back
        tool-call envelopes — see DSML TOOLCALLING.batch.2.b).
      * vLLM emits ``None``.
      * SGLang emits ``""``.

    All three express the same semantic and should compare equal."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def canonical(d: dict[str, Any]) -> str:
    """Canonical JSON for diffing: sorted keys, no whitespace, with empty-string ↔ None
    normalization applied to parser text fields."""
    if "normal_text" in d:
        d = {**d, "normal_text": _normalize_normal_text(d["normal_text"])}
    if "reasoning_text" in d:
        d = {**d, "reasoning_text": _normalize_normal_text(d["reasoning_text"])}
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def decode_arguments(args: Any) -> Any:
    """Decode a tool-call `arguments` field to a comparable Python value.

    Parsers surface `arguments` slightly differently: JSON-encoded
    string (Dynamo, vLLM), already-parsed dict (some SGLang paths),
    or the truncated raw string verbatim on malformed input. Decode
    JSON if it parses; otherwise return as-is so the canonical-diff
    surfaces the mismatch rather than swallowing it.
    """
    if not args:
        return {}
    if not isinstance(args, str):
        return args
    try:
        return json.loads(args)
    except (json.JSONDecodeError, TypeError):
        return args


def decode_stream_calls(
    stream_calls: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    calls = []
    for _, call in sorted(stream_calls.items()):
        if not call.get("name") and not call.get("arguments"):
            continue
        calls.append(
            {
                "name": call.get("name") or "",
                "arguments": decode_arguments(call.get("arguments") or ""),
            }
        )
    return calls


def linkify_text_html(text: str) -> str:
    """Escape plain text and turn embedded URLs into anchors."""

    parts: list[str] = []
    last = 0
    for match in URL_RE.finditer(text):
        raw_url = match.group(0)
        url = raw_url.rstrip(TRAILING_URL_PUNCTUATION)
        if not url:
            continue
        parts.append(html_lib.escape(text[last : match.start()]))
        href = html_lib.escape(url, quote=True)
        parts.append(
            f'<a href="{href}" target="_blank" rel="noopener noreferrer">'
            f"{html_lib.escape(url)}</a>"
        )
        parts.append(html_lib.escape(raw_url[len(url) :]))
        last = match.end()
    parts.append(html_lib.escape(text[last:]))
    return "".join(parts)


def parity_cell_class(marker: str) -> str:
    if marker == "—":
        return "missing"
    if marker == "n/a":
        return "na"
    if marker == "D":
        return "donly"
    if "!" in marker:
        return "err"
    if "↯" in marker:
        return "leak"
    if "?" in marker:
        return "research"
    if marker == "=":
        return "ok"
    return "documented"


def ref_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def build_parity_tooltip_html(
    *,
    head: str,
    description: str | None = None,
    input_label: str | None = None,
    input_html: str | None = None,
    output_sections: list[tuple[str, str]] | None = None,
    divergent_reasons: str | None = None,
    leak_label: str | None = None,
    leak_text: str | None = None,
    extra_sections: list[tuple[str, str]] | None = None,
    refs: list[tuple[str, Any]] | None = None,
) -> str:
    """Build the hover popup used by parser and reasoning parity tables.

    The section order is intentional and shared across the two tables:
    context, input, outputs, divergence explanation, leak warning, harness
    details, and finally provenance refs.
    """

    parts = ['<div class="ttip">']
    if head:
        parts.append(f'<div class="ttip-head">{html_lib.escape(head)}</div>')
    if description:
        parts.append(f'<pre class="ttip-pre">{html_lib.escape(description)}</pre>')

    def add_section(label: str, body_html: str) -> None:
        parts.append(f'<div class="ttip-section">{html_lib.escape(label)}:</div>')
        parts.append(f'<pre class="ttip-pre">{body_html}</pre>')

    if input_label and input_html is not None:
        add_section(input_label, input_html)

    for label, body_html in output_sections or []:
        add_section(label, body_html)

    if divergent_reasons:
        add_section("Divergent reasons", linkify_text_html(divergent_reasons))

    if leak_label and leak_text:
        add_section(leak_label, linkify_text_html(leak_text))

    for label, body_html in extra_sections or []:
        add_section(label, body_html)

    for label, value in refs or []:
        if value:
            add_section(label, linkify_text_html(ref_text(value)))

    parts.append("</div>")
    return "".join(parts)
