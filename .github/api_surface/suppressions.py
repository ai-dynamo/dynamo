# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Suppression waivers for the API surface validator.

``validate`` treats removing (or prematurely removing) a ``stable`` symbol as a
broken promise. Some of those are intentional -- a never-really-public symbol, a
removal eng explicitly signed off on -- and a waiver file lets a human silence a
specific finding without weakening the check for everything else.

The waiver file (``.github/api-surface/suppressions.yaml``) is
git-tracked and auditable:

```yaml
suppressions:
  - id: "python:dynamo.runtime._internal.helper"   # exact symbol id
    reason: "Never public; leaked into the stub by mistake."
  - pattern: "helm:platform:components.planner.*"   # fnmatch glob on id
    surface: helm                                    # optional surface guard
    reason: "Planner chart values are alpha; churn expected."
    until: "1.3.0"                                   # waiver expires after 1.3.0
```

Matching is exact-id first, then ``fnmatch`` glob on ``pattern``; an optional
``surface`` further narrows a rule. ``until`` bounds the waiver to releases up to
and including that version, so a stale waiver self-expires rather than hiding a
real regression forever.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from api_surface.models import SURFACES

_FIELDS = {"id", "pattern", "surface", "reason", "until"}


def version_key(release: str) -> tuple[int, ...]:
    """Sortable key for a dotted release string ('1.2.0' -> (1, 2, 0)).

    Non-numeric or empty components degrade to ``0`` so a malformed value sorts
    low rather than raising mid-validation. Shared with the validator so waiver
    expiry and removal-target ordering compare versions the same way.
    """
    parts: list[int] = []
    for chunk in str(release).strip().split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


@dataclass
class Suppression:
    """One waiver rule. Exactly one of ``id`` / ``pattern`` should be set.

    Attributes:
        id: Exact symbol id to waive.
        pattern: ``fnmatch`` glob over symbol ids (used when ``id`` is empty).
        surface: Optional surface guard; when set the rule only matches symbols
            on that surface.
        reason: Human justification (required).
        until: Last release the waiver covers; empty means it never expires.
    """

    id: str = ""
    pattern: str = ""
    surface: str = ""
    reason: str = ""
    until: str = ""

    def matches(self, symbol_id: str, surface: str = "", release: str = "") -> bool:
        """Whether this rule waives ``symbol_id`` (optionally at ``release``)."""
        if self.surface and surface and self.surface != surface:
            return False
        if release and self.until and version_key(release) > version_key(self.until):
            return False
        if self.id:
            return symbol_id == self.id
        if self.pattern:
            return fnmatch.fnmatchcase(symbol_id, self.pattern)
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary, dropping empty optional fields."""
        out: dict[str, Any] = {}
        for key in ("id", "pattern", "surface", "reason", "until"):
            value = getattr(self, key)
            if value:
                out[key] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Suppression:
        """Deserialize from a single waiver mapping."""
        unknown = sorted(str(key) for key in data if key not in _FIELDS)
        if unknown:
            raise ValueError(f"unknown suppression field(s): {', '.join(unknown)}")
        for field_name in _FIELDS:
            value = data.get(field_name, "")
            if not isinstance(value, str):
                raise ValueError(f"suppression {field_name} must be a string")
        exact_id = data.get("id", "").strip()
        pattern = data.get("pattern", "").strip()
        if bool(exact_id) == bool(pattern):
            raise ValueError("suppression must set exactly one of id or pattern")
        reason = data.get("reason", "").strip()
        if not reason:
            raise ValueError("suppression reason must not be empty")
        surface = data.get("surface", "").strip()
        if surface and surface not in SURFACES:
            raise ValueError(f"invalid suppression surface: {surface}")
        return cls(
            id=exact_id,
            pattern=pattern,
            surface=surface,
            reason=reason,
            until=data.get("until", "").strip(),
        )


@dataclass
class SuppressionSet:
    """A loaded set of waiver rules with a single match entry point."""

    rules: list[Suppression] = field(default_factory=list)

    def matching(
        self, symbol_id: str, surface: str = "", release: str = ""
    ) -> Suppression | None:
        """Return the first rule waiving ``symbol_id``, or ``None``."""
        for rule in self.rules:
            if rule.matches(symbol_id, surface=surface, release=release):
                return rule
        return None

    def is_suppressed(
        self, symbol_id: str, surface: str = "", release: str = ""
    ) -> bool:
        """True when any rule waives ``symbol_id`` (optionally at ``release``)."""
        return self.matching(symbol_id, surface=surface, release=release) is not None


def load_suppressions(path: str | Path) -> SuppressionSet:
    """Load the waiver file, returning an empty set when it is absent.

    Accepts either a top-level ``suppressions:`` list or a bare list document.
    """
    path = Path(path)
    if not path.exists():
        return SuppressionSet()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"failed to load suppression manifest: {error}") from error
    raw = data.get("suppressions", data) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("suppression manifest must contain a list")
    if not all(isinstance(rule, dict) for rule in raw):
        raise ValueError("every suppression must be a mapping")
    rules = [Suppression.from_dict(rule) for rule in raw]
    return SuppressionSet(rules=rules)


def default_suppressions_path(root: str | Path = ".") -> Path:
    """Return the project-local waiver path."""
    return Path(root) / ".github" / "api-surface" / "suppressions.yaml"
