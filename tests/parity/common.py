# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contract types + canonical-JSON diff for parity (parser) impls.

Every impl wrapper (parser-mode and e2e-mode) returns ParseResult so the
harness can diff results uniformly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


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


def _normalize_normal_text(v: Any) -> Any:
    """Treat any whitespace-only or None value as equivalent for comparison.

    Engines disagree on the carrier for "no narration":
      * Dynamo emits ``""`` (or ``"\\n"`` carried through between back-to-back
        tool-call envelopes — see DSML PARSER.batch.2.b).
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
    normalization applied to `normal_text`."""
    if "normal_text" in d:
        d = {**d, "normal_text": _normalize_normal_text(d["normal_text"])}
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
