# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pool selector matching, shared by every selector-keyed config table.

A **selector** is a slash-separated path matched against a pool's identity,
``"<participant_id>/<sub_type>"`` -- today ``"<k8s namespace>/<deployment>/<sub_type>"``,
though nothing here depends on that depth.

- ``*`` matches exactly one segment.
- ``**`` matches any number of segments, including none. It is only special as
  a *whole* segment: inside one, as in ``gpt-oss-**``, it is an ordinary glob
  star and still matches within that single segment only.
- A selector shorter than the pool path covers everything beneath it, so
  ``prod/chat`` selects every pool of that deployment while
  ``prod/chat/prefill`` selects one.

When several selectors match, the most specific wins, ranked by how many
segments are named exactly and then by depth -- independent of the order
entries appear in the config file. Ties fall back to declaration order.

Named ``pool_selectors`` rather than ``selectors`` on purpose: the latter
would shadow the Python standard library's ``selectors`` module for any tool or
script whose working directory puts this package on ``sys.path``, breaking
``asyncio`` and anything else that imports it.

This lives apart from any one table so pool priorities
(:mod:`~dynamo.global_planner.priority`) and declared GPU costs
(:mod:`~dynamo.global_planner.gpu_cost`) cannot drift into subtly different
matching rules.
"""

from __future__ import annotations

import fnmatch
from typing import Optional, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

_WILDCARD_CHARS = "*?["


class PoolSelector(BaseModel):
    """Base for a config entry scoped to a set of pools by path pattern."""

    model_config = ConfigDict(extra="forbid")

    selector: str = Field(
        description=(
            "Slash-separated path matched against '<participant_id>/<sub_type>'. "
            "'*' matches exactly one segment, '**' matches any number, and a "
            "selector shorter than the pool path covers everything beneath it."
        )
    )

    @model_validator(mode="after")
    def _validate_selector(self) -> "PoolSelector":
        segments = self.selector.split("/")
        if not segments or any(not seg for seg in segments):
            raise ValueError(
                f"selector {self.selector!r} must be a slash-separated path with "
                f"no empty segments"
            )
        return self

    @property
    def _segments(self) -> tuple[str, ...]:
        return tuple(self.selector.split("/"))

    @property
    def _pattern(self) -> tuple[str, ...]:
        """Declared segments, implicitly covering everything beneath them.

        A selector shorter than a pool path should select every pool under it,
        which is exactly "match the declared segments, then anything" -- so a
        trailing ``**`` is appended unless one is already there.
        """
        segments = self._segments
        if segments[-1] == "**":
            return segments
        return segments + ("**",)

    @staticmethod
    def _match_segments(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
        """Glob a segment path, where a whole ``**`` segment spans any number.

        Wildcards never span a ``/`` unless the segment is exactly ``**``, so
        ``a/*/prefill`` matches ``a/b/prefill`` but not ``a/b/c/prefill``,
        while ``a/**/prefill`` matches both. A ``**`` embedded in a longer
        segment is just a glob star within that segment.
        """
        if not pattern:
            return not path
        head, rest = pattern[0], pattern[1:]
        if head == "**":
            # Try consuming 0, 1, 2 ... segments here.
            return any(
                PoolSelector._match_segments(rest, path[i:])
                for i in range(len(path) + 1)
            )
        if not path:
            return False
        if not fnmatch.fnmatchcase(path[0], head):
            return False
        return PoolSelector._match_segments(rest, path[1:])

    def matches(self, participant_id: str, sub_type: str) -> bool:
        """Whether this selector covers the pool ``participant_id``/``sub_type``."""
        path = tuple(participant_id.split("/")) + (sub_type,)
        return self._match_segments(self._pattern, path)

    @property
    def specificity(self) -> tuple[int, int]:
        """Sort key ranking selectors most-specific first (smaller wins).

        Ranked by how many segments are named exactly, then by depth. So
        ``prod/chat/prefill`` beats ``prod/chat``, which beats ``prod/*``,
        which beats ``**``.
        """
        exact = sum(
            1 for seg in self._segments if not any(c in seg for c in _WILDCARD_CHARS)
        )
        return (-exact, -len(self._segments))


EntryT = TypeVar("EntryT", bound=PoolSelector)


def order_by_specificity(entries: Sequence[EntryT]) -> list[EntryT]:
    """Order entries most-specific first, ties broken by declaration order.

    Ordering once at construction keeps resolution deterministic and
    independent of how the config file happened to be written.
    """
    return [
        entry
        for _, entry in sorted(
            enumerate(entries), key=lambda pair: (pair[1].specificity, pair[0])
        )
    ]


def reject_duplicate_selectors(entries: Sequence[PoolSelector]) -> None:
    """Raise if two entries declare the same selector.

    Only one of them could ever take effect, so a duplicate is a config
    mistake rather than a precedence question.
    """
    seen: set[str] = set()
    for entry in entries:
        if entry.selector in seen:
            raise ValueError(
                f"duplicate selector {entry.selector!r}: merge the entries, "
                f"since only one of them could ever take effect"
            )
        seen.add(entry.selector)


def first_match(
    ordered: Sequence[EntryT], participant_id: str, sub_type: str
) -> Optional[EntryT]:
    """Most specific entry covering this pool, or ``None``."""
    for entry in ordered:
        if entry.matches(participant_id, sub_type):
            return entry
    return None
