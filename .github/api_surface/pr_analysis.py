# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-PR API-surface impact analysis.

Given the surface snapshots at a PR's base and head refs, classify the diff
into the buckets a reviewer or automated PR comment consumer cares about:

- **breaking** -- a breaking-type diff change (removal, incompatible signature,
  rename, relocation, or served-version removal) on a symbol that was ``stable`` and not already
  deprecated at the base. These are the broken-promise changes the stability
  policy says require a deprecation cycle first.
- **informational** -- the same breaking-type changes on ``preview`` /
  ``experimental`` / already-``deprecated`` symbols. Allowed by the tiers, shown
  so the author knows what moved.
- **additive** -- ``added`` symbols (new public API; always safe).
- **relocated** -- moves of non-stable or already-deprecated symbols.

The analysis is pure: it consumes two already-built :class:`SurfaceSnapshot`
objects via :func:`gather_changes` and joins each change against the base/head
symbol stability. Building the snapshots (slow, git-bound) is the caller's job,
so this module stays fast and unit-testable without a checkout. This is the
read seam the package docstring promises -- drive it with merge-base vs merge
commit to get a PR's surface impact.
"""

from __future__ import annotations

import json
from typing import Any

from api_surface.models import SurfaceChange, SurfaceSnapshot
from api_surface.results import OperationResult
from api_surface.snapshot import gather_changes
from api_surface.suppressions import SuppressionSet

# Output formats for :func:`render_impact`.
IMPACT_FORMATS = frozenset({"md", "slack", "json"})

# Diff change types that remove or alter an existing promise. A rename or move
# removes the old import path just as surely as a deletion does.
_BREAKING_DIFF_TYPES = frozenset(
    {
        "removed",
        "signature_changed",
        "served_version_removed",
        "renamed_function",
        "relocated",
    }
)


def _is_protected(
    change: SurfaceChange, base: dict[str, Any], head: dict[str, Any]
) -> bool:
    """True when the change hits a ``stable``, non-deprecated symbol.

    Looks the symbol up in the base snapshot first (where a removed/changed
    symbol lived), then the head. An unknown symbol is treated as protected so
    the analyzer fails safe -- it flags rather than silently drops a change it
    cannot tier.
    """
    sym = base.get(change.id) or head.get(change.id)
    if sym is None:
        return True
    return sym.stability not in {"preview", "experimental"} and not sym.deprecated


def analyze_impact(
    base: SurfaceSnapshot,
    head: SurfaceSnapshot,
    *,
    repo_path: str | None = None,
    release: str = "",
    include_markers: bool = False,
    suppressions: SuppressionSet | None = None,
) -> OperationResult:
    """Classify the surface delta between two snapshots into reviewer buckets.

    Args:
        base: Snapshot at the PR's merge-base (the prior state).
        head: Snapshot at the PR's merge commit (the new state).
        repo_path: Optional tree for the native-marker scan; ``None`` (default)
            keeps the analysis code-diff only, which is what a per-PR impact
            wants.
        release: Release the head belongs to (forwarded to ``gather_changes``).
        include_markers: Forwarded to ``gather_changes``; default ``False``.
        suppressions: Waiver rules; a breaking change matching a rule (bounded
            by ``until`` relative to ``release``) moves to ``waived`` and no
            longer trips ``has_breaking``.

    Returns:
        :class:`OperationResult` whose ``data`` carries ``base_ref`` /
        ``head_ref``, a ``counts`` map, and the ``breaking`` /
        ``informational`` / ``additive`` / ``relocated`` / ``waived`` change
        lists (each a list of :meth:`SurfaceChange.to_dict`; ``waived`` entries
        also carry the matching rule's ``reason``).
        ``metadata['has_breaking']`` is the gate signal and reflects the
        post-waiver breaking list. ``coverage_gaps`` is carried through so a
        missing surface is never misread as mass removals.
    """
    change_result = gather_changes(
        base,
        head,
        repo_path=repo_path,
        release=release,
        include_markers=include_markers,
    )
    changes = [
        SurfaceChange.from_dict(c) for c in change_result.data.get("changes", [])
    ]
    base_sym = {s.id: s for s in base.symbols}
    head_sym = {s.id: s for s in head.symbols}

    breaking: list[SurfaceChange] = []
    informational: list[SurfaceChange] = []
    additive: list[SurfaceChange] = []
    relocated: list[SurfaceChange] = []

    for change in changes:
        if change.change_type == "added":
            additive.append(change)
        elif change.change_type in _BREAKING_DIFF_TYPES:
            if _is_protected(change, base_sym, head_sym):
                breaking.append(change)
            elif change.change_type == "relocated":
                relocated.append(change)
            else:
                informational.append(change)
        else:
            # Any change type outside the known diff vocabulary (e.g. a marker
            # event) is surfaced as informational rather than dropped.
            informational.append(change)

    waived: list[dict[str, Any]] = []
    if suppressions is not None:
        kept: list[SurfaceChange] = []
        for change in breaking:
            rule = suppressions.matching(
                change.id, surface=change.surface, release=release
            )
            if rule is None:
                kept.append(change)
                continue
            waived.append({**change.to_dict(), "reason": rule.reason})
        breaking = kept

    data: dict[str, Any] = {
        "base_ref": base.ref,
        "head_ref": head.ref,
        "counts": {
            "breaking": len(breaking),
            "informational": len(informational),
            "additive": len(additive),
            "relocated": len(relocated),
            "waived": len(waived),
            "total": len(changes),
        },
        "breaking": [c.to_dict() for c in breaking],
        "informational": [c.to_dict() for c in informational],
        "additive": [c.to_dict() for c in additive],
        "relocated": [c.to_dict() for c in relocated],
        "waived": waived,
        "coverage_gaps": change_result.data.get("coverage_gaps", []),
    }
    return OperationResult(
        data=data,
        errors=list(change_result.errors),
        metadata={
            "has_breaking": bool(breaking),
            "change_count": len(changes),
            "base_ref": base.ref,
            "head_ref": head.ref,
        },
    )


def _surface_counts(changes: list[dict[str, Any]]) -> str:
    """``rust 12, python 4`` style breakdown for a change list."""
    counts: dict[str, int] = {}
    for c in changes:
        counts[c.get("surface", "?")] = counts.get(c.get("surface", "?"), 0) + 1
    return ", ".join(f"{s} {n}" for s, n in sorted(counts.items())) or "none"


def _change_line(c: dict[str, Any], *, bullet: str) -> str:
    """One reviewer-facing bullet for a single change."""
    surface = c.get("surface", "?")
    sid = c["id"]
    ctype = c["change_type"]
    if ctype in {"relocated", "renamed_function"} and c.get("relocated_to"):
        return f"{bullet} `{surface}` {sid} -> {c['relocated_to']}"
    detail = c.get("summary", "")
    if ctype == "signature_changed" and not detail:
        detail = f"`{c.get('from_signature', '')}` -> `{c.get('to_signature', '')}`"
    suffix = f": {detail}" if detail else ""
    return f"{bullet} `{surface}` {sid} ({ctype}){suffix}"


def _render_md(data: dict[str, Any]) -> str:
    """Markdown impact report, sized for a PR comment."""
    counts = data["counts"]
    lines = [
        "# API Surface Impact",
        "",
        f"**Base:** `{data['base_ref'] or '?'}`  **Head:** `{data['head_ref'] or '?'}`",
        "",
        "**Summary**",
        "",
        f"- Breaking: {counts['breaking']}",
        f"- Informational: {counts['informational']}",
        f"- Added: {counts['additive']}",
        f"- Relocated: {counts['relocated']}",
        f"- Waived: {counts.get('waived', 0)}",
    ]
    if data["coverage_gaps"]:
        lines.append(f"- Coverage Gaps: {', '.join(data['coverage_gaps'])}")

    lines += ["", f"## Breaking Changes ({counts['breaking']})", ""]
    if data["breaking"]:
        lines.append(
            "Stable public API removed or changed without a deprecation cycle."
        )
        lines.append("")
        lines += [_change_line(c, bullet="-") for c in data["breaking"]]
    else:
        lines.append("None. No stable public API was removed or altered.")

    if data["informational"]:
        lines += ["", f"## Informational ({counts['informational']})", ""]
        lines.append(
            "Preview / experimental / already-deprecated; allowed by the tiers."
        )
        lines.append("")
        lines += [_change_line(c, bullet="-") for c in data["informational"]]

    if data.get("waived"):
        lines += ["", f"## Waived ({counts.get('waived', len(data['waived']))})", ""]
        lines.append(
            "Breaking changes covered by a reviewed waiver in suppressions.yaml."
        )
        lines.append("")
        for c in data["waived"]:
            reason = c.get("reason", "")
            suffix = f" — {reason}" if reason else ""
            lines.append(f"{_change_line(c, bullet='-')}{suffix}")

    if data["relocated"]:
        lines += ["", f"## Relocated ({counts['relocated']})", ""]
        lines += [_change_line(c, bullet="-") for c in data["relocated"]]

    if data["additive"]:
        lines += ["", f"## Added ({counts['additive']})", ""]
        lines.append(f"New public API across {_surface_counts(data['additive'])}.")

    return "\n".join(lines) + "\n"


def _render_slack(data: dict[str, Any]) -> str:
    """Slack-native impact report: ``*bold*`` headers + ``•`` bullets."""
    counts = data["counts"]
    lines = [
        f"*API Surface Impact:* `{data['base_ref'] or '?'}` to `{data['head_ref'] or '?'}`",
        (
            f"Breaking: {counts['breaking']}  Informational: {counts['informational']}  "
            f"Added: {counts['additive']}  Relocated: {counts['relocated']}"
        ),
    ]
    if data.get("waived"):
        lines.append(
            f":memo: Waived: {counts.get('waived', len(data['waived']))} "
            "breaking change(s) covered by suppressions."
        )
    if data["breaking"]:
        lines.append("")
        lines.append(":warning: *Breaking* (stable API removed or changed):")
        lines += [_change_line(c, bullet="•") for c in data["breaking"]]
    else:
        lines.append(":white_check_mark: No breaking changes to stable API.")
    if data["relocated"]:
        lines.append("")
        lines.append("*Relocated:*")
        lines += [_change_line(c, bullet="•") for c in data["relocated"]]
    return "\n".join(lines)


def render_impact(data: dict[str, Any], fmt: str = "md") -> str:
    """Render :func:`analyze_impact` output in ``md`` / ``slack`` / ``json``.

    Raises:
        ValueError: when ``fmt`` is not in :data:`IMPACT_FORMATS`.
    """
    if fmt not in IMPACT_FORMATS:
        raise ValueError(
            f"unknown format {fmt!r}; expected one of {sorted(IMPACT_FORMATS)}"
        )
    if fmt == "json":
        return json.dumps(data, indent=2, sort_keys=True)
    if fmt == "slack":
        return _render_slack(data)
    return _render_md(data)
