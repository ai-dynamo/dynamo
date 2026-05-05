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
    """Treat empty string and None as equivalent.

    Different impls emit different sentinels for "the model produced no
    narration text" — some return `""`, others return `None` (e.g.,
    OpenAI-shape `content: null`). Both express the same semantic, so
    canonical comparison collapses them.
    """
    if v == "" or v is None:
        return None
    return v


def canonical(d: dict[str, Any]) -> str:
    """Canonical JSON for diffing: sorted keys, no whitespace, with empty-string ↔ None
    normalization applied to `normal_text`."""
    if "normal_text" in d:
        d = {**d, "normal_text": _normalize_normal_text(d["normal_text"])}
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def diff(a: ParseResult, b: ParseResult) -> bool:
    """Return True if the two results are equivalent under canonical compare."""
    return canonical(a.to_dict()) == canonical(b.to_dict())
